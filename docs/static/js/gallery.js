(function() {
  var rows = [
    { track: document.getElementById('gallery-track-1'), reverse: true },
    { track: document.getElementById('gallery-track-2'), reverse: false }
  ];

  if (!rows[0].track) return;

  if (typeof GALLERY_IMAGES === 'undefined' || GALLERY_IMAGES.length === 0) {
    return;
  }

  var shuffled = GALLERY_IMAGES.slice().sort(function() { return Math.random() - 0.5; });
  var n = Math.floor(shuffled.length / 2);
  var groups = [
    shuffled.slice(0, n),
    shuffled.slice(n, 2 * n)
  ];

  var basePath = 'static/images/camouflage_example/';
  var resizedPath = 'static/images/camouflage_example_resized/';
  var labeledPath = 'static/images/camouflage_example_resized_labeled/';
  var speedPerImage = 12;
  var SWITCH_DELAY = 1;

  rows.forEach(function(row, i) {
    var images = groups[i];
    var fragment = document.createDocumentFragment();

    for (var dup = 0; dup < 2; dup++) {
      for (var j = 0; j < images.length; j++) {
        var filename = images[j];

        var item = document.createElement('div');
        item.className = 'gallery-item';

        // Original image
        var img = document.createElement('img');
        img.src = basePath + filename;
        img.alt = '';
        img.loading = 'lazy';
        img.className = 'gallery-img';
        item.appendChild(img);

        // Overlay container
        var overlay = document.createElement('div');
        overlay.className = 'gallery-overlay';

        var overlayResized = document.createElement('img');
        overlayResized.className = 'overlay-img overlay-resized';
        overlayResized.src = resizedPath + filename;
        overlayResized.alt = '';

        var overlayLabeled = document.createElement('img');
        overlayLabeled.className = 'overlay-img overlay-labeled';
        overlayLabeled.src = labeledPath + filename;
        overlayLabeled.alt = '';

        overlay.appendChild(overlayResized);
        overlay.appendChild(overlayLabeled);
        item.appendChild(overlay);

        // Hover logic
        var timer = null;
        var captionDetail = document.getElementById('captionDetail');
        var captionBg = document.getElementById('captionBg');
        var captionObj = document.getElementById('captionObj');
        var leaveDelay = null;

        item.addEventListener('mouseenter', function() {
          var fname = this._filename;
          var ov = this.querySelector('.gallery-overlay');
          var labeled = this.querySelector('.overlay-labeled');
          ov.classList.add('visible');
          labeled.classList.remove('show');

          if (captionDetail) clearTimeout(leaveDelay);
          var labels = (typeof GALLERY_LABELS !== 'undefined' && GALLERY_LABELS[fname])
            ? GALLERY_LABELS[fname] : { background: '', object: '' };
          if (captionBg) captionBg.textContent = labels.background || '';
          if (captionObj) captionObj.textContent = labels.object || '';
          if (captionDetail) captionDetail.classList.add('visible');

          timer = setTimeout(function() {
            labeled.classList.add('show');
          }, SWITCH_DELAY);
        });

        item.addEventListener('mouseleave', function() {
          var ov = this.querySelector('.gallery-overlay');
          var labeled = this.querySelector('.overlay-labeled');
          if (timer) {
            clearTimeout(timer);
            timer = null;
          }
          ov.classList.remove('visible');
          labeled.classList.remove('show');

          if (captionDetail) {
            leaveDelay = setTimeout(function() {
              captionDetail.classList.remove('visible');
            }, 150);
          }
        });

        item._filename = filename;
        fragment.appendChild(item);
      }
    }

    row.track.appendChild(fragment);

    var duration = images.length * speedPerImage;
    row.track.style.animationDuration = duration + 's';
    if (row.reverse) {
      row.track.classList.add('gallery-track-reverse');
    }
  });
})();
