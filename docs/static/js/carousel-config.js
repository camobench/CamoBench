// Fill in image filenames (relative to static/images/) and captions
// Use \n in captions for line breaks
// captionWidth: max width as percentage of container (0-100), default 80 if omitted
const CAROUSEL_IMAGES = [
  { src: "dataset_overview.jpg", caption: "A complete data sample in CamoBench, with the background-object metadata (left top), inputs for three paradigms (left bottom), and generated images with human scores (right).", captionWidth: 70 },
  { src: "camobench_statistics.jpg", caption: "CamoBench includes 512 background-object combinations, each with inputs for three generative paradigms. We generate 17,920 images using 35 models and collect human evaluations for 13,824 images from 27 models.", captionWidth: 85 },
  { src: "taxo_backgrounds.jpg", caption: "Visual examples of the 4 background types. Each column shows representative examples of one type, with 32 samples for each type.", captionWidth: 60 },
  { src: "taxo_objects.jpg", caption: "Visual examples of the hidden-object taxonomy. The 12 object sub-categories are grouped under 4 high-level types: Textual Elements, Geometry & Shapes, Rigid Objects, and Non-Rigid Objects.", captionWidth: 75 },
  { src: "hidden_object_statistics.jpg", caption: "The hierarchical composition of the 128 hidden objects. The inner ring represents the four super-categories, each with 32 objects, and the outer ring indicates the instance counts for the 12 sub-categories.", captionWidth: 82 },
  { src: "dataset_statistics.jpg", caption: "The distribution of the 512 background-object combinations. The x-axis groups the 12 object sub-categories by super-category. The stacked bars demonstrate the equal distribution of the four background textures within each sub-category.", captionWidth: 95 }
];
