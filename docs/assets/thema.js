// Helles und dunkles Design umschalten. Ohne eigene Wahl gilt die Einstellung des Systems.
// Die Wahl gilt für alle Seiten des Workshops und bleibt im Browser gespeichert.
// Damit beim Laden nichts aufblitzt, setzt jede Seite das gespeicherte Design schon im <head>.
(() => {
  const SCHLUESSEL = "ki-workshop-thema";
  const wurzel = document.documentElement;
  const systemDunkel = matchMedia("(prefers-color-scheme: dark)");
  const aktuell = () => wurzel.dataset.theme || (systemDunkel.matches ? "dark" : "light");
  const knoepfe = document.querySelectorAll(".thema-knopf");

  const beschriften = () => {
    const text = aktuell() === "dark" ? "Helles Design einschalten" : "Dunkles Design einschalten";
    knoepfe.forEach(knopf => {
      knopf.setAttribute("aria-label", text);
      knopf.title = text;
    });
  };

  knoepfe.forEach(knopf => knopf.addEventListener("click", () => {
    const neu = aktuell() === "dark" ? "light" : "dark";
    wurzel.dataset.theme = neu;
    try { localStorage.setItem(SCHLUESSEL, neu); } catch (e) { /* ohne Speicher gilt die Wahl nur für diese Seite */ }
    beschriften();
  }));

  systemDunkel.addEventListener("change", beschriften);
  beschriften();
})();
