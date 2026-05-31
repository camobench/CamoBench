import streamlit as st
import os
import glob
import pandas as pd
import json
import random
import time
import shutil
from collections import defaultdict
import streamlit.components.v1 as components
from PIL import Image
import threading
import io

# ================= Configuration =================

IMAGE_ROOT_DIR = "./data/generated_images/images_pending"

CSV_PATH_LOGIC = "./data/prompts/dataset.csv"
CSV_PATH_DISPLAY = "./data/prompts/dataset.csv"
# CSV_PATH_LOGIC = "./data/iaa/iaa_dataset.csv"
# CSV_PATH_DISPLAY = "./data/iaa/iaa_dataset.csv"

SAVE_DIR = "./data/score"
COLLECTION_DIR = "./data/generated_images/collections"

# Auto-save interval (seconds)
AUTO_SAVE_INTERVAL = 60

# [New] Annotation target rounds (1, 2, or 3)
# 1: Each image only needs 1 annotation (stop after score1)
# 2: Each image needs 2 annotations (stop after score1, score2)
# 3: Each image needs 3 annotations (stop after score1, score2, score3)
TARGET_ROUND = 2

# ===========================================

# Initialize directories
for d in [SAVE_DIR, COLLECTION_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

st.set_page_config(layout="wide", page_title=f"Benchmark Annotator v6.2 (Target: {TARGET_ROUND})")

# --- 1. Data Manager ---

class DataManager:
    def __init__(self):
        # Modification: Load two CSV files separately
        self.csv_logic = self.load_csv(CSV_PATH_LOGIC)
        self.csv_display = self.load_csv(CSV_PATH_DISPLAY)

        # file_map structure: { "ModelName": { "image_id": "full_path" } }
        self.file_map = self.scan_images()
        # data structure: { "ModelName": { "image_id": { "Metric": {"score1":.., "score2":.., "score3":..} } } }
        self.annotations = self.load_all_jsonl()

    @st.cache_data
    def load_csv(_self, file_path):
        """Generic CSV loading function"""
        try:
            if not os.path.exists(file_path):
                st.error(f"Cannot find file: {file_path}")
                return {}
            df = pd.read_csv(file_path)
            # Ensure ID is a string
            df['id'] = df['id'].astype(str)
            return df.set_index('id').to_dict('index')
        except Exception as e:
            st.error(f"CSV read error ({file_path}): {e}")
            return {}

    def scan_images(self):
        """Scan images and parse model names"""
        mapping = defaultdict(dict)
        if not os.path.exists(IMAGE_ROOT_DIR):
            return mapping
        
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        
        # Iterate subdirectories
        for folder_name in os.listdir(IMAGE_ROOT_DIR):
            full_folder_path = os.path.join(IMAGE_ROOT_DIR, folder_name)
            if not os.path.isdir(full_folder_path):
                continue
            
            model_name = folder_name

            # Scan files
            images = glob.glob(os.path.join(full_folder_path, "*.*"))
            for img_path in images:
                if img_path.lower().endswith(valid_exts):
                    # Parse image_id
                    filename = os.path.basename(img_path)
                    name_no_ext = os.path.splitext(filename)[0]
                    try:
                        # Assume id is at the end, e.g. xxx_123.png -> 123
                        img_id = name_no_ext.split('_')[-1]
                    except:
                        img_id = "unknown"
                    
                    mapping[model_name][img_id] = img_path
        return mapping

    def load_all_jsonl(self):
        """Read all .jsonl files in the score directory"""
        data = defaultdict(lambda: defaultdict(dict))
        
        jsonl_files = glob.glob(os.path.join(SAVE_DIR, "*.jsonl"))
        for fpath in jsonl_files:
            filename = os.path.basename(fpath)
            model_name = filename.replace(".jsonl", "")
            
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        entry = json.loads(line)
                        img_id = str(entry.get("id"))
                        # Store entire row data in memory
                        # Filter out id field, store only metric data
                        metrics = {k: v for k, v in entry.items() if k != "id"}
                        data[model_name][img_id] = metrics
            except Exception as e:
                st.error(f"Reading {filename} failed: {e}")
        return data

    def save_all_jsonl(self, current_data):
        """Write data from memory back to corresponding jsonl files"""
        # current_data structure: {ModelName: {ID: {Metric: ...}}}
        timestamp = time.time()
        
        for model_name, images_dict in current_data.items():
            file_path = os.path.join(SAVE_DIR, f"{model_name}.jsonl")
            
            # Prepare the list of lines to write
            lines_to_write = []
            # Sort by ID before writing, keep it tidy (optional)
            sorted_ids = sorted(images_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)
            
            for img_id in sorted_ids:
                entry = {"id": img_id}
                entry.update(images_dict[img_id])
                lines_to_write.append(json.dumps(entry, ensure_ascii=False))
            
            # Write file (overwrite mode)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines_to_write))
        
        return timestamp

# --- 2. Core Logic: Queue and Grouping ---

def generate_queue(dm: DataManager, metric_name):
    """
    Generate annotation queue based on Metric and TARGET_ROUND.
    """
    tasks = []  # element: {'model': m, 'id': i, 'path': p, 'group_key': k}
    
    # Determine grouping basis
    # group_mode = "random"
    group_mode = "background"
    if metric_name in ["Illusion Fidelity", "Semantic Consistency – Illusion Shape Consistency"]:
        group_mode = "object"
    elif metric_name == "Semantic Consistency – Scene Consistency":
        group_mode = "background"
    
    # Iterate in-memory data
    for model_name, img_dict in dm.annotations.items():
        for img_id, metrics in img_dict.items():
            # Check if physical image file exists
            img_path = dm.file_map.get(model_name, {}).get(img_id)
            if not img_path: continue
            
            if metric_name not in metrics: continue
            
            # === [Modification Start] Logic Change ===
            scores = metrics[metric_name] # {'score1': 5, 'score2': None, ...}

            # Calculate current annotation count (number of non-None)
            filled_count = 0
            for k in ['score1', 'score2', 'score3']:
                if scores.get(k) is not None:
                    filled_count += 1

            # Calculate how many more annotations needed to reach TARGET_ROUND
            # e.g.: Target=1, Filled=0 -> need 1
            #       Target=1, Filled=1 -> need 0 (skip)
            #       Target=2, Filled=0 -> need 2
            needed_count = TARGET_ROUND - filled_count

            # Ensure not less than 0, and total cannot exceed 3 slots
            # (theoretically needed_count is correct, but guard against config errors)
            if needed_count > 0:
                # Get grouping key
                csv_info = dm.csv_logic.get(img_id, {})
                
                if group_mode == "object":
                    key = csv_info.get("Camouflaged Object Description", "Unknown")
                elif group_mode == "background":
                    key = csv_info.get("Background Description", "Unknown")
                else:
                    key = "All"
                
                # Add this image to queue needed_count times
                for _ in range(needed_count):
                    tasks.append({
                        "model": model_name,
                        "id": img_id,
                        "path": img_path,
                        "group_key": key
                    })
            # === [Modification End] ===
    
    # Start grouping and shuffling
    grouped_tasks = defaultdict(list)
    for t in tasks:
        grouped_tasks[t['group_key']].append(t)
    
    final_queue = []
    
    # 1. Randomize group order
    group_keys = list(grouped_tasks.keys())
    random.shuffle(group_keys)
    
    for key in group_keys:
        group_items = grouped_tasks[key]
        # 2. Randomize images within group
        random.shuffle(group_items)
        final_queue.extend(group_items)
        
    return final_queue

def get_next_empty_slot_key(score_dict):
    """Return the first key in 'score1', 'score2', 'score3' that is None"""
    for k in ['score1', 'score2', 'score3']:
        if score_dict.get(k) is None:
            return k
    return None

def get_last_filled_slot_key(score_dict):
    """Return the last non-None key (for undo)"""
    for k in ['score3', 'score2', 'score1']:
        if score_dict.get(k) is not None:
            return k
    return None

@st.cache_resource(show_spinner=False, max_entries=36)
def get_resized_image(img_path, max_edge):
    try:
        img = Image.open(img_path)
        w, h = img.size
        if w > h:
            new_w = max_edge
            new_h = int(h * (max_edge / w))
        else:
            new_h = max_edge
            new_w = int(w * (max_edge / h))
        resized = img.resize((new_w, new_h))
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None
    
# --- 3. Streamlit Main Program ---

def main():
    if 'dm' not in st.session_state:
        st.session_state['dm'] = DataManager()
    
    dm = st.session_state['dm']
    
    # Session State initialization
    if 'last_save_time' not in st.session_state: st.session_state['last_save_time'] = time.time()
    if 'queue' not in st.session_state: st.session_state['queue'] = []
    if 'current_metric' not in st.session_state: st.session_state['current_metric'] = ""
    if 'history_stack' not in st.session_state: st.session_state['history_stack'] = []
    if 'group_transition' not in st.session_state: st.session_state['group_transition'] = False
    
    # === Sidebar ===
    with st.sidebar:
        st.title("🧩 Annotation Tool")
        st.caption(f"Target Rounds: {TARGET_ROUND}")
        
        # Prominent save button
        if st.button("💾 SAVE ALL DATA", type="primary", use_container_width=True):
            dm.save_all_jsonl(dm.annotations)
            st.session_state['last_save_time'] = time.time()
            st.toast("✅ All data has been saved to score/*.jsonl")

        st.divider()
        
        # Select metric
        metrics = [
            "Illusion Fidelity", 
            "Overall Visual Quality", 
            "Semantic Consistency – Scene Consistency", 
            "Semantic Consistency – Illusion Shape Consistency"
        ]
        selected_metric = st.selectbox("Select current metric", [""] + metrics)
        
        # When switching Metric, regenerate queue
        if selected_metric != st.session_state['current_metric']:
            st.session_state['current_metric'] = selected_metric
            if selected_metric:
                st.session_state['queue'] = generate_queue(dm, selected_metric)
                st.session_state['history_stack'] = [] # Clear undo stack when switching metric
                st.rerun()

        # Display remaining progress
        if selected_metric:
            q_len = len(st.session_state['queue'])
            st.metric("Remaining Tasks (Total Slots)", q_len)
            if q_len == 0:
                st.success(f"Current metric completed {TARGET_ROUND} rounds of annotations!")

    # Auto-save check
    if time.time() - st.session_state['last_save_time'] > AUTO_SAVE_INTERVAL:
        dm.save_all_jsonl(dm.annotations)
        st.session_state['last_save_time'] = time.time()
        # st.toast("System automatically saving...")

    # If no metric selected or queue is empty
    if not selected_metric or not st.session_state['queue']:
        st.info("Please select an annotation metric on the left, or all images for the current metric have been annotated.")
        return

    # === Get Current Task ===
    current_task = st.session_state['queue'][0] # Take queue head, don't pop yet, pop after scoring
    
    model_name = current_task['model']
    img_id = current_task['id']
    img_path = current_task['path']
    # Note: group_key here comes from the English version dataset.csv
    group_key = current_task['group_key'] 
    
    # Get display info for UI presentation
    csv_info_display = dm.csv_display.get(img_id, {})

    # Group transition notification (balloons)
    if st.session_state['group_transition']:
        st.balloons()
        st.toast(f"⚠️ Entering new scene/object category: {group_key}", icon="🎈")
        st.session_state['group_transition'] = False

    # === UI Layout ===
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.image(img_path, use_container_width=True)
        st.caption(f"Model: {model_name} | ID: {img_id}")

    with c2:
        try:
            # pil_img = Image.open(img_path)
            
            img_128 = get_resized_image(img_path, 128)
            img_64 = get_resized_image(img_path, 64)
            img_32 = get_resized_image(img_path, 32)
            
            tc1, tc2 = st.columns(2)
            with tc1:
                st.image(img_path, caption="1. Original Image", use_container_width=True)
                st.image(img_64, caption="3. Resized to 64px", use_container_width=True)
            with tc2:
                st.image(img_128, caption="2. Resized to 128px", use_container_width=True)
                st.image(img_32, caption="4. Resized to 32px", use_container_width=True)
                
        except Exception as e:
            st.error(f"Failed to load auxiliary views: {e}")
        
        st.divider()

        st.markdown("### Metadata")
        bg_desc = csv_info_display.get('Background Description', 'NA')
        obj_desc = csv_info_display.get('Camouflaged Object Description', 'NA')
        
        st.info(f"**Background:** {bg_desc}")
        st.info(f"**Object:** {obj_desc}")
        
        st.divider()
        st.markdown(f"**Current Metric:** `{selected_metric}`")
        
        # Determine valid button range
        valid_keys = []
        if selected_metric == "Illusion Fidelity": valid_keys = ["1", "2", "3", "4", "5"]
        elif selected_metric == "Overall Visual Quality": valid_keys = ["1", "2", "3", "4", "5"]
        elif selected_metric == "Semantic Consistency – Scene Consistency": valid_keys = ["0", "1"]
        elif selected_metric == "Semantic Consistency – Illusion Shape Consistency": valid_keys = ["1", "2", "3"]

    # === Action Handlers ===
    def submit_score(score_val):
        # 1. Find the slot to fill (score1, score2, or score3)
        current_metrics = dm.annotations[model_name][img_id][selected_metric]
        slot_key = get_next_empty_slot_key(current_metrics)
        
        if slot_key:
            # 2. Write data
            dm.annotations[model_name][img_id][selected_metric][slot_key] = score_val
            
            # 3. Record history (for undo)
            popped_task = st.session_state['queue'].pop(0) # Remove queue head
            st.session_state['history_stack'].append({
                "model": model_name,
                "id": img_id,
                "metric": selected_metric,
                "slot": slot_key,
                "task": popped_task
            })
            
            # 4. Check if next image belongs to a new group
            if st.session_state['queue']:
                next_task = st.session_state['queue'][0]
                if next_task['group_key'] != popped_task['group_key']:
                    st.session_state['group_transition'] = True
            
            # Auto-save (for safety, update in-memory state each time, write to disk on timer)
        else:
            st.error("Data anomaly: this image's metric is fully scored but still appears in the queue.")
            st.session_state['queue'].pop(0) # Force remove

    def undo_last():
        if st.session_state['history_stack']:
            last_action = st.session_state['history_stack'].pop()
            # Restore data to None
            dm.annotations[last_action['model']][last_action['id']][last_action['metric']][last_action['slot']] = None
            # Put task back to queue head
            st.session_state['queue'].insert(0, last_action['task'])
            st.toast("Undid last step")

    def collect_image():
        # Copy file
        target_name = f"{model_name}_{img_id}{os.path.splitext(img_path)[1]}"
        target_path = os.path.join(COLLECTION_DIR, target_name)
        try:
            shutil.copy2(img_path, target_path)
            st.toast(f"❤️ Collected: {target_name}")
        except Exception as e:
            st.error(f"Failed to collect: {e}")

    # === Render Buttons ===
    st.markdown("---")
    btn_cols = st.columns(len(valid_keys) + 2) # +2 for Undo and Collect

    # Scoring buttons
    for i, key in enumerate(valid_keys):
        with btn_cols[i]:
            st.button(
                key, 
                key=f"btn_{key}", 
                on_click=submit_score, 
                args=(key,), 
                use_container_width=True, 
                type="primary"
            )
            
    # Undo button
    with btn_cols[-2]:
        st.button("⬅️ withdraw", key="btn_undo", on_click=undo_last, use_container_width=True)
        
    # Collect button (implicitly triggered via Space key, or click here)
    with btn_cols[-1]:
        st.button("❤️ collect", key="btn_collect", on_click=collect_image, use_container_width=True)

    # === JS Keyboard Listener ===
    js_code = f"""
    <script>
    const validKeys = {json.dumps(valid_keys)};
    
    // [Core Fix 1]: Find the real function reference from the previous round in global variables, precise removal to prevent infinite stacking!
    if (window.parent._my_keydown) {{
        window.parent.document.removeEventListener('keydown', window.parent._my_keydown);
        window.parent.document.removeEventListener('keyup', window.parent._my_keyup);
    }}

    let overlay = window.parent.document.getElementById('score-overlay');
    if (!overlay) {{
        overlay = window.parent.document.createElement('div');
        overlay.id = 'score-overlay';
        overlay.style.cssText = "position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.8);color:white;font-size:20vw;display:flex;justify-content:center;align-items:center;z-index:999999;pointer-events:none;visibility:hidden;font-family:Arial;";
        window.parent.document.body.appendChild(overlay);
    }}

    // [Core Fix 2]: Store current function entity in global variable window.parent._my_keydown
    window.parent._my_keydown = function(e) {{
        if (e.repeat) return; // Intercept physical multi-tap caused by holding the key too long
        
        if (validKeys.includes(e.key)) {{
            overlay.innerText = e.key;
            overlay.style.visibility = 'visible';
        }} else if (e.code === "Space") {{
            overlay.innerText = "❤️";
            overlay.style.visibility = 'visible';
        }}
    }};

    window.parent._my_keyup = function(e) {{
        overlay.style.visibility = 'hidden';
        let buttons = Array.from(window.parent.document.querySelectorAll('button'));
        
        if (validKeys.includes(e.key)) {{
            let targetBtn = buttons.find(b => b.innerText === e.key);
            if (targetBtn) targetBtn.click();
        }}
        else if (e.key === "ArrowLeft") {{
            let undoBtn = buttons.find(b => b.innerText.includes("withdraw"));
            if (undoBtn) undoBtn.click();
        }}
        else if (e.code === "Space") {{
            let colBtn = buttons.find(b => b.innerText.includes("collect"));
            if (colBtn) colBtn.click();
        }}
    }};

    // Bind current listener
    window.parent.document.addEventListener('keydown', window.parent._my_keydown);
    window.parent.document.addEventListener('keyup', window.parent._my_keyup);
    </script>
    """

    paths_to_preload = []
    if len(st.session_state['queue']) > 3:
        for i in range(1, min(3, len(st.session_state['queue']))):
            paths_to_preload.append(st.session_state['queue'][i]['path'])

    def preload_worker(paths):
        for p in paths:
            # get_original_image(p)
            get_resized_image(p, 128)
            get_resized_image(p, 64)
            get_resized_image(p, 32)

    if paths_to_preload:
        threading.Thread(target=preload_worker, args=(paths_to_preload,), daemon=True).start()

    components.html(js_code, height=0)

if __name__ == "__main__":
    main()