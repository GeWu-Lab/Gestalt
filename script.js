'use strict';

// Track sections in reading order; the mobile directory stays horizontally scrollable.
const navigation = [...document.querySelectorAll('.toc-link')];
const sections = navigation.map(link => document.getElementById(link.hash.slice(1)));
const directory = document.querySelector('.sidebar');
const readingLayout = document.querySelector('.reading-layout');
let scheduled = false;
function updateNavigation() {
  scheduled = false;
  const compact = matchMedia('(max-width: 900px)').matches;
  // Reveal in place when the reader reaches the paper, without shifting its layout.
  const showDirectory = readingLayout.getBoundingClientRect().top <= (compact ? 76 : 120);
  directory.classList.toggle('is-visible', showDirectory);
  directory.inert = !showDirectory;
  directory.setAttribute('aria-hidden', String(!showDirectory));
  const threshold = Math.max(compact ? 100 : 90, innerHeight * .25);
  let current = '';
  for (const section of sections) {
    if (section && section.getBoundingClientRect().top <= threshold) current = `#${section.id}`;
  }
  for (const link of navigation) {
    if (link.hash === current) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
}
function scheduleNavigation() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(updateNavigation);
}
addEventListener('scroll', scheduleNavigation, { passive: true });
addEventListener('resize', scheduleNavigation);
addEventListener('load', scheduleNavigation);
updateNavigation();

// Hover and keyboard focus preview a zone. Tap again or press Escape to restore the full figure.
const canvas = document.querySelector('.architecture-canvas');
const zoneButtons = [...document.querySelectorAll('.zone-card')];
const panels = [...document.querySelectorAll('.zoom-panel')];
const canHover = matchMedia('(hover: hover)');
let pinned = null;
let hovered = null;
let focused = null;
function renderZone() {
  const active = hovered || focused || pinned || 'all';
  canvas.dataset.zone = active;
  canvas.setAttribute('aria-label', active === 'all' ? 'Complete Gestalt architecture' : `Enlarged Interplay Zone ${active === 'one' ? 'I: Bottleneck Interplay' : 'II: Full Multimodal Interplay'}`);
  for (const button of zoneButtons) {
    button.setAttribute('aria-pressed', String(pinned === button.dataset.zone));
    button.classList.toggle('is-previewed', active === button.dataset.zone);
  }
  for (const panel of panels) panel.setAttribute('aria-hidden', String(panel.dataset.detail !== active));
}
for (const button of zoneButtons) {
  const zone = button.dataset.zone;
  button.addEventListener('pointerenter', event => {
    if (!canHover.matches || event.pointerType === 'touch') return;
    hovered = zone;
    renderZone();
  });
  button.addEventListener('pointerleave', () => { hovered = null; renderZone(); });
  button.addEventListener('focus', () => {
    if (button.matches(':focus-visible')) { focused = zone; renderZone(); }
  });
  button.addEventListener('blur', () => { focused = null; renderZone(); });
  button.addEventListener('click', () => {
    pinned = pinned === zone ? null : zone;
    hovered = null;
    focused = null;
    renderZone();
  });
}
function resetZone() {
  pinned = null;
  hovered = null;
  focused = null;
  renderZone();
}
document.querySelector('.architecture-explorer').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); resetZone(); }
});
renderZone();
