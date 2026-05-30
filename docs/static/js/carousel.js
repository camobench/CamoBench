(function() {
  var slidesEl = document.getElementById('carouselSlides');
  var captionEl = document.getElementById('carouselCaption');
  var navEl = document.getElementById('carouselNav');
  var carousel = document.getElementById('imgCarousel');

  if (!slidesEl || !navEl || typeof CAROUSEL_IMAGES === 'undefined' || CAROUSEL_IMAGES.length === 0) return;

  var images = CAROUSEL_IMAGES.filter(function(item) { return item.src; });
  if (images.length === 0) return;

  var basePath = 'static/images/';
  var current = 0;

  // Build slides
  images.forEach(function(item, i) {
    var slide = document.createElement('div');
    slide.className = 'carousel-slide' + (i === 0 ? ' active' : '');
    var img = document.createElement('img');
    img.src = basePath + item.src;
    img.alt = item.caption || '';
    slide.appendChild(img);
    slidesEl.appendChild(slide);
  });

  // Build nav: triangle left + dots + triangle right
  var prevTri = document.createElement('button');
  prevTri.className = 'carousel-tri carousel-tri-prev';
  prevTri.setAttribute('aria-label', 'Previous image');
  prevTri.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="18,4 4,12 18,20"/></svg>';
  prevTri.addEventListener('click', prev);
  navEl.appendChild(prevTri);

  var dotsContainer = document.createElement('div');
  dotsContainer.className = 'carousel-dots';
  images.forEach(function(_, i) {
    var dot = document.createElement('button');
    dot.className = 'carousel-dot' + (i === 0 ? ' active' : '');
    dot.setAttribute('aria-label', 'Image ' + (i + 1));
    dot.addEventListener('click', function() { goTo(i); });
    dotsContainer.appendChild(dot);
  });
  navEl.appendChild(dotsContainer);

  var nextTri = document.createElement('button');
  nextTri.className = 'carousel-tri carousel-tri-next';
  nextTri.setAttribute('aria-label', 'Next image');
  nextTri.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="6,4 20,12 6,20"/></svg>';
  nextTri.addEventListener('click', next);
  navEl.appendChild(nextTri);

  function setCaption(item) {
    if (!captionEl) return;
    var lines = (item.caption || '').split('\n');
    captionEl.innerHTML = '';
    lines.forEach(function(line, i) {
      if (i > 0) captionEl.appendChild(document.createElement('br'));
      captionEl.appendChild(document.createTextNode(line));
    });
    var w = item.captionWidth;
    captionEl.style.maxWidth = (w != null ? w : 80) + '%';
  }

  setCaption(images[0]);
  captionEl.style.marginLeft = 'auto';
  captionEl.style.marginRight = 'auto';

  function goTo(index) {
    if (index < 0 || index >= images.length) return;
    var slides = slidesEl.querySelectorAll('.carousel-slide');
    var dots = navEl.querySelectorAll('.carousel-dot');
    slides[current].classList.remove('active');
    dots[current].classList.remove('active');
    current = index;
    slides[current].classList.add('active');
    dots[current].classList.add('active');
    setCaption(images[current]);
  }

  function next() { goTo((current + 1) % images.length); }
  function prev() { goTo((current - 1 + images.length) % images.length); }

  document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowLeft') prev();
    if (e.key === 'ArrowRight') next();
  });
})();
