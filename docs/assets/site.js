// Lesefortschritt, aktiver Abschnitt im Inhaltsverzeichnis, vergrößerte Bilder und Ausdruck
(() => {
  const balken = document.querySelector(".fortschritt div");
  const text = document.querySelector(".text");
  if (balken && text) {
    const aktualisieren = () => {
      const box = text.getBoundingClientRect();
      const anteil = Math.min(1, Math.max(0, -box.top / Math.max(1, box.height - innerHeight)));
      balken.style.width = (anteil * 100).toFixed(2) + "%";
    };
    addEventListener("scroll", aktualisieren, { passive: true });
    addEventListener("resize", aktualisieren);
    aktualisieren();
  }

  const links = [...document.querySelectorAll(".verzeichnis ol a")];
  if (links.length && "IntersectionObserver" in window) {
    const sichtbar = new Map();
    const markieren = () => {
      const oben = links.find(a => sichtbar.get(decodeURIComponent(a.hash.slice(1))));
      if (!oben) return;
      links.forEach(a => a.classList.toggle("aktiv", a === oben));
    };
    const beobachter = new IntersectionObserver(eintraege => {
      eintraege.forEach(e => sichtbar.set(e.target.id, e.isIntersecting));
      markieren();
    }, { rootMargin: "-80px 0px -55% 0px" });
    links.forEach(a => {
      const ziel = document.getElementById(decodeURIComponent(a.hash.slice(1)));
      if (ziel) beobachter.observe(ziel);
    });
  }

  // Auf schmalen Bildschirmen das Inhaltsverzeichnis zunächst zuklappen
  const verzeichnis = document.querySelector(".verzeichnis details");
  if (verzeichnis && matchMedia("(max-width: 980px)").matches) verzeichnis.open = false;

  document.querySelectorAll(".text figure img").forEach(bild => {
    bild.addEventListener("click", () => {
      const ebene = document.createElement("div");
      ebene.className = "zoom";
      ebene.innerHTML = `<img src="${bild.src}" alt="">`;
      ebene.addEventListener("click", () => ebene.remove());
      addEventListener("keydown", function weg(e) { if (e.key === "Escape") { ebene.remove(); removeEventListener("keydown", weg); } });
      document.body.append(ebene);
    });
  });

  // Beim Drucken alle aufklappbaren Kästen öffnen, damit das Handout vollständig ist
  addEventListener("beforeprint", () => document.querySelectorAll(".text details").forEach(d => { d.open = true; }));
})();
