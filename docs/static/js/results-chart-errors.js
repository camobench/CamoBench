(function() {
  var container = document.getElementById('resultsErrorChartContainer');
  if (!container || typeof RESULTS_ERROR_DATA === 'undefined') return;

  // Group models by paradigm
  var paradigms = {};
  RESULTS_ERROR_DATA.forEach(function(m) {
    if (!paradigms[m.paradigm]) paradigms[m.paradigm] = [];
    paradigms[m.paradigm].push(m);
  });

  // Build checkboxes
  var checklistEl = document.getElementById('resultsErrorChecklist');
  var paradigmOrder = ['T2I', 'Editing', 'CondGen'];
  paradigmOrder.forEach(function(p) {
    if (!paradigms[p]) return;
    var group = document.createElement('div');
    group.className = 'results-paradigm-group';

    var titleRow = document.createElement('label');
    titleRow.className = 'results-paradigm-toggle';
    titleRow.innerHTML = '<input type="checkbox" class="paradigm-checkbox" data-paradigm="' + p + '"> <span>' + p + '</span>';
    group.appendChild(titleRow);

    var modelList = document.createElement('div');
    modelList.className = 'results-model-list';

    paradigms[p].forEach(function(m) {
      var item = document.createElement('label');
      item.className = 'results-model-item';
      if (m.name.indexOf('Average') === 0 || m.name.indexOf('Average (') === 0) {
        item.classList.add('results-model-average');
      }
      item.innerHTML = '<input type="checkbox" class="model-checkbox" data-key="' + p + '::' + m.name + '"> ' + m.name;
      modelList.appendChild(item);
    });

    group.appendChild(modelList);
    checklistEl.appendChild(group);
  });

  // Default selection
  var defaultKeys = [
    'T2I::Average (T2I)',
    'Editing::Average (Editing)',
    'CondGen::Average (CondGen)'
  ];
  defaultKeys.forEach(function(key) {
    var cb = checklistEl.querySelector('.model-checkbox[data-key="' + key + '"]');
    if (cb) cb.checked = true;
  });
  // Sync paradigm checkboxes
  paradigmOrder.forEach(function(p) {
    var all = checklistEl.querySelectorAll('.model-checkbox[data-key^="' + p + '::"]');
    var allChecked = true;
    all.forEach(function(cb) { if (!cb.checked) allChecked = false; });
    var pc = checklistEl.querySelector('.paradigm-checkbox[data-paradigm="' + p + '"]');
    if (pc) pc.checked = allChecked;
  });

  checklistEl.addEventListener('change', function(e) {
    if (e.target.classList.contains('paradigm-checkbox')) {
      var p = e.target.dataset.paradigm;
      var checked = e.target.checked;
      checklistEl.querySelectorAll('.model-checkbox[data-key^="' + p + '::"]').forEach(function(cb) {
        cb.checked = checked;
      });
      updateChart();
    } else if (e.target.classList.contains('model-checkbox')) {
      var p = e.target.dataset.key.split('::')[0];
      var all = checklistEl.querySelectorAll('.model-checkbox[data-key^="' + p + '::"]');
      var allChecked = true;
      all.forEach(function(cb) { if (!cb.checked) allChecked = false; });
      checklistEl.querySelector('.paradigm-checkbox[data-paradigm="' + p + '"]').checked = allChecked;
      updateChart();
    }
  });

  // Chart.js colors
  var barColors = [
    'rgba(37, 99, 235, 0.85)',
    'rgba(16, 185, 129, 0.85)',
    'rgba(245, 158, 11, 0.85)',
    'rgba(239, 68, 68, 0.85)',
    'rgba(139, 92, 246, 0.85)',
    'rgba(236, 72, 153, 0.85)',
    'rgba(20, 184, 166, 0.85)',
    'rgba(249, 115, 22, 0.85)'
  ];

  var barBorders = [
    'rgba(37, 99, 235, 1)',
    'rgba(16, 185, 129, 1)',
    'rgba(245, 158, 11, 1)',
    'rgba(239, 68, 68, 1)',
    'rgba(139, 92, 246, 1)',
    'rgba(236, 72, 153, 1)',
    'rgba(20, 184, 166, 1)',
    'rgba(249, 115, 22, 1)'
  ];

  // Init chart
  if (typeof Chart === 'undefined') {
    container.innerHTML = '<p style="color:red;">Chart.js failed to load.</p>';
    return;
  }

  try {
    var canvas = document.getElementById('resultsErrorChart');
    var ctx = canvas.getContext('2d');
    var chart = new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { maxRotation: 45, minRotation: 0, font: { size: 11 } }
        },
        y: {
          beginAtZero: true,
          ticks: { font: { size: 12 } }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { usePointStyle: true, padding: 16, font: { size: 12 } }
        }
      }
    }
  });

    requestAnimationFrame(function() {
      requestAnimationFrame(function() {
        chart.resize();
      });
    });
    window.addEventListener('resize', function() { chart.resize(); });
  } catch(e) {
    container.innerHTML = '<p style="color:red;">Chart init error: ' + e.message + '</p>';
    return;
  }

  function updateChart() {
    var selected = [];
    checklistEl.querySelectorAll('.model-checkbox:checked').forEach(function(cb) {
      selected.push(cb.dataset.key);
    });

    var datasets = RESULTS_ERROR_METRICS.map(function(metric, i) {
      return {
        label: metric,
        data: [],
        backgroundColor: barColors[i],
        borderColor: barBorders[i],
        borderWidth: 1
      };
    });

    var labels = [];

    RESULTS_ERROR_DATA.forEach(function(m) {
      var key = m.paradigm + '::' + m.name;
      if (selected.indexOf(key) === -1) return;
      labels.push(m.name);
      m.values.forEach(function(v, i) {
        datasets[i].data.push(v);
      });
    });

    chart.data.labels = labels;
    chart.data.datasets = datasets;
    chart.update();
  }

  updateChart();
})();
