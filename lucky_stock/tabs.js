document.addEventListener("DOMContentLoaded", () => {
  const tabs = Array.from(document.querySelectorAll(".tablinks"));
  const contents = Array.from(document.querySelectorAll(".tabcontent"));
  const navTabs = document.querySelector(".nav-tabs");
  const menuToggle = document.querySelector(".menu-toggle");
  const validTabs = tabs.map((tab) => tab.dataset.tab);

  function openTab(tabName, updateHash = true) {
    const nextTab = validTabs.includes(tabName) ? tabName : "home";

    contents.forEach((content) => {
      content.classList.toggle("active", content.id === nextTab);
    });

    tabs.forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === nextTab);
    });

    if (navTabs) navTabs.classList.remove("show");
    if (updateHash) history.replaceState(null, "", `#${nextTab}`);
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => openTab(tab.dataset.tab));
  });

  if (menuToggle && navTabs) {
    menuToggle.addEventListener("click", () => navTabs.classList.toggle("show"));
  }

  window.addEventListener("hashchange", () => openTab(location.hash.slice(1), false));
  openTab(location.hash.slice(1) || "home", false);
});
