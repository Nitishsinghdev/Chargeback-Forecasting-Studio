(function () {
  "use strict";

  document.querySelectorAll("[data-plotly]").forEach(function (el) {
    try {
      var spec = JSON.parse(el.getAttribute("data-plotly"));
      var layout = spec.layout || {};
      layout.autosize = true;
      layout.margin = layout.margin || { l: 48, r: 24, t: 40, b: 48 };
      layout.paper_bgcolor = "rgba(0,0,0,0)";
      layout.plot_bgcolor = "rgba(0,0,0,0)";
      Plotly.newPlot(el, spec.data, layout, { responsive: true, displayModeBar: true });
    } catch (e) {
      console.error("Plotly render failed", e);
    }
  });

  var parseBtn = document.getElementById("btn-parse-events");
  if (parseBtn) {
    parseBtn.addEventListener("click", function () {
      var textarea = document.getElementById("event_text_batch");
      var out = document.getElementById("parse-preview");
      if (!textarea || !out) return;
      var token = parseBtn.getAttribute("data-session");
      var csrf = parseBtn.getAttribute("data-csrf");
      fetch("/s/" + token + "/api/parse-events", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": csrf,
        },
        body: JSON.stringify({ text: textarea.value }),
      })
        .then(function (r) {
          return r.json();
        })
        .then(function (data) {
          if (data.error) {
            out.innerHTML = '<div class="alert alert-danger">' + data.error + "</div>";
            return;
          }
          out.innerHTML = data.html || "<pre>" + JSON.stringify(data.events, null, 2) + "</pre>";
        })
        .catch(function () {
          out.innerHTML = '<div class="alert alert-danger">Parse request failed.</div>';
        });
    });
  }
})();
