document.addEventListener("DOMContentLoaded", function () {
  const switchBox = document.getElementById("publication-view-switch");
  if (!switchBox) return;

  const buttons = Array.from(switchBox.querySelectorAll("button[data-view]"));
  const views = {
    list: document.getElementById("publication-view-list"),
    cards: document.getElementById("publication-view-cards"),
  };
  const storageKey = "publication-view";

  function applyView(view) {
    buttons.forEach(function (button) {
      button.classList.toggle("active", button.dataset.view === view);
    });

    Object.keys(views).forEach(function (key) {
      if (views[key]) {
        views[key].classList.toggle("active", key === view);
      }
    });

    try {
      localStorage.setItem(storageKey, view);
    } catch (e) {}
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      applyView(button.dataset.view);
    });
  });

  let preferred = "list";
  try {
    preferred = localStorage.getItem(storageKey) || "list";
  } catch (e) {}
  if (!views[preferred]) preferred = "list";
  applyView(preferred);
});
