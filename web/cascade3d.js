
import * as THREE from "/vendor/three.module.min.js";

const TIERS = [
  { key: "tier1_exact", label: "Exact", detail: "UTR and total agree", color: "#3E9E6B" },
  { key: "tier2_fuzzy", label: "Fuzzy", detail: "within tolerance", color: "#7CC79A" },
  { key: "tier3_subset_sum", label: "Arithmetic", detail: "reconstructed", color: "#4E93C9" },
  { key: "tier4_adjudicated", label: "Adjudicated", detail: "model asked", color: "#D8A24A" },
  { key: "tier5_exception", label: "Exception", detail: "queued for a human", color: "#D0563F" },
];

const VOID = 0x0c0d0b;
const SLAB = 0x1e211d;
const RULE = 0x3a4038;
const COLS = 22;
const SPACING = 0.52;
const STEP_Y = 1.45;
const STEP_Z = 2.7;

function makeLabel(text, sub, align = "left") {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 128;
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  g.textAlign = align;
  const x = align === "left" ? 8 : c.width - 8;
  g.fillStyle = "#EDE8DE";
  g.font = "600 46px -apple-system, Segoe UI, Roboto, Helvetica, sans-serif";
  g.fillText(text, x, 52);
  if (sub) {
    g.fillStyle = "#8E958A";
    g.font = "400 32px -apple-system, Segoe UI, Roboto, Helvetica, sans-serif";
    g.fillText(sub, x, 96);
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.scale.set(4, 1, 1);
  return sprite;
}

function fitDistance(box, camera, dir, pad = 1.03) {
  const centre = box.getCenter(new THREE.Vector3());
  const right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
  const up = new THREE.Vector3().crossVectors(right, dir).normalize();
  const tanY = Math.tan((camera.fov * Math.PI) / 360);
  const tanX = tanY * camera.aspect;

  let distance = 0;
  const { min, max } = box;
  for (let i = 0; i < 8; i++) {
    const corner = new THREE.Vector3(
      i & 1 ? max.x : min.x,
      i & 2 ? max.y : min.y,
      i & 4 ? max.z : min.z
    ).sub(centre);
    const along = corner.dot(dir);
    distance = Math.max(
      distance,
      along + Math.abs(corner.dot(up)) / tanY,
      along + Math.abs(corner.dot(right)) / tanX
    );
  }
  return distance * pad;
}

function orbit(camera, target, dom, onInteract, radius, theta, phi) {
  const state = { radius, theta, phi, dragging: false, lx: 0, ly: 0 };

  function apply() {
    state.phi = Math.max(0.18, Math.min(1.45, state.phi));
    state.radius = Math.max(radius * 0.45, Math.min(radius * 2.4, state.radius));
    camera.position.set(
      target.x + state.radius * Math.sin(state.phi) * Math.sin(state.theta),
      target.y + state.radius * Math.cos(state.phi),
      target.z + state.radius * Math.sin(state.phi) * Math.cos(state.theta)
    );
    camera.lookAt(target);
  }

  function down(e) {
    state.dragging = true;
    const p = e.touches ? e.touches[0] : e;
    state.lx = p.clientX;
    state.ly = p.clientY;
    onInteract();
  }
  function move(e) {
    if (!state.dragging) return;
    const p = e.touches ? e.touches[0] : e;
    state.theta -= (p.clientX - state.lx) * 0.007;
    state.phi -= (p.clientY - state.ly) * 0.007;
    state.lx = p.clientX;
    state.ly = p.clientY;
    apply();
    if (e.touches) e.preventDefault();
  }
  const up = () => (state.dragging = false);

  dom.addEventListener("pointerdown", down);
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
  dom.addEventListener("touchstart", down, { passive: true });
  dom.addEventListener("touchmove", move, { passive: false });
  dom.addEventListener("touchend", up);
  dom.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      state.radius += e.deltaY * 0.018;
      onInteract();
      apply();
    },
    { passive: false }
  );

  apply();
  return { state, apply };
}

export function mountCascade(container, lines, onSelect) {
  container.innerHTML = "";

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (err) {
    container.innerHTML =
      '<p class="note" style="padding:20px 0">This browser has no WebGL, so the cascade view is ' +
      "unavailable. Every figure it shows is in the tables below.</p>";
    return { dispose() {} };
  }

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const width = container.clientWidth || 900;
  const height = Math.max(460, Math.min(780, Math.round(width * 0.74)));

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(width, height);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.style.display = "block";
  renderer.domElement.style.touchAction = "none";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 200);
  const target = new THREE.Vector3(0, -2.6, 0);

  scene.add(new THREE.HemisphereLight(0xdfe6f0, 0x14170f, 1.15));
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const key = new THREE.DirectionalLight(0xfff2dc, 1.5);
  key.position.set(9, 18, 11);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  scene.add(key);

  const groups = TIERS.map((t) => lines.filter((l) => l.tier === t.key));
  const slabW = COLS * SPACING + 1.4;

  const cubeGeo = new THREE.BoxGeometry(1, 1, 1);
  const picks = [];
  const animated = [];
  const disposables = [cubeGeo];
  const ROW_GAP = 0.62;
  const LABEL_GAP = 2.2;

  TIERS.forEach((tier, i) => {
    const y = -i * STEP_Y;
    const z = i * STEP_Z - (TIERS.length - 1) * STEP_Z * 0.5;
    const items = groups[i];

    const rows = Math.max(1, Math.ceil(items.length / COLS));
    const depth = Math.max(2.5, rows * ROW_GAP + 0.8);

    const slabGeo = new THREE.BoxGeometry(slabW, 0.22, depth);
    disposables.push(slabGeo);
    const slab = new THREE.Mesh(slabGeo, new THREE.MeshLambertMaterial({ color: SLAB }));
    slab.position.set(0, y, z);
    slab.receiveShadow = true;
    scene.add(slab);

    const edgeGeo = new THREE.BoxGeometry(slabW, 0.03, 0.03);
    disposables.push(edgeGeo);
    const edge = new THREE.Mesh(edgeGeo, new THREE.MeshBasicMaterial({ color: RULE }));
    edge.position.set(0, y + 0.12, z - depth / 2 + 0.05);
    scene.add(edge);

    const label = makeLabel(tier.label, `${items.length} \u00b7 ${tier.detail}`);
    label.position.set(-slabW / 2 - LABEL_GAP, y + 0.7, z);
    label.scale.set(3.9, 0.98, 1);
    scene.add(label);

    const color = new THREE.Color(tier.color);
    items.forEach((line, n) => {
      const col = n % COLS;
      const row = Math.floor(n / COLS);
      const mag = Math.abs(line.amount_paise) / 1000 + 1;
      const h = 0.22 + Math.log10(mag) * 0.34;

      const mesh = new THREE.Mesh(cubeGeo, new THREE.MeshLambertMaterial({ color }));
      mesh.scale.set(0.36, h, 0.36);
      const restY = y + 0.11 + h / 2;
      mesh.position.set(
        (col - (COLS - 1) / 2) * SPACING,
        restY,
        z + (row - (rows - 1) / 2) * ROW_GAP
      );
      mesh.castShadow = true;
      mesh.userData = { line, restY, base: color.clone() };
      scene.add(mesh);
      picks.push(mesh);

      if (!reduced) {
        animated.push({ mesh, restY, lift: 11 + Math.random() * 6, delay: i * 0.16 + n * 0.006 });
      }
    });
  });

  const bounds = new THREE.Box3();
  scene.traverse((o) => {
    if (o.isMesh) bounds.expandByObject(o);
  });
  bounds.expandByPoint(new THREE.Vector3(-slabW / 2 - LABEL_GAP - 2.2, bounds.min.y, bounds.min.z));
  bounds.getCenter(target);

  const span = bounds.getSize(new THREE.Vector3());
  const reach = Math.max(span.x, span.y, span.z) * 0.5;
  const THETA0 = -0.6;
  const PHI0 = 1.06;
  const viewDir = new THREE.Vector3(
    Math.sin(PHI0) * Math.sin(THETA0),
    Math.cos(PHI0),
    Math.sin(PHI0) * Math.cos(THETA0)
  );
  const fit = fitDistance(bounds, camera, viewDir);

  scene.fog = new THREE.Fog(VOID, fit * 0.85, fit * 2.6);
  camera.far = fit * 4;
  camera.updateProjectionMatrix();

  key.position.set(span.x * 0.5, fit * 0.8, span.z * 0.9);
  key.target.position.copy(target);
  scene.add(key.target);
  const sc = key.shadow.camera;
  const pad = reach * 1.25;
  sc.left = -pad;
  sc.right = pad;
  sc.top = pad;
  sc.bottom = -pad;
  sc.near = 0.5;
  sc.far = fit * 3;
  sc.updateProjectionMatrix();

  animated.forEach((a) => {
    a.mesh.position.y = a.restY + a.lift;
  });

  const tooltip = document.createElement("div");
  tooltip.className = "tip";
  container.style.position = "relative";
  container.appendChild(tooltip);

  let idle = !reduced;
  const controls = orbit(camera, target, renderer.domElement, () => (idle = false), fit, THETA0, PHI0);

  const ray = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let hovered = null;

  function pick(e) {
    const r = renderer.domElement.getBoundingClientRect();
    pointer.x = ((e.clientX - r.left) / r.width) * 2 - 1;
    pointer.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    ray.setFromCamera(pointer, camera);
    return ray.intersectObjects(picks, false)[0];
  }

  renderer.domElement.addEventListener("pointermove", (e) => {
    const hit = pick(e);
    if (hovered && (!hit || hit.object !== hovered)) {
      hovered.material.color.copy(hovered.userData.base);
      hovered = null;
      tooltip.style.display = "none";
    }
    if (hit && hit.object !== hovered) {
      hovered = hit.object;
      hovered.material.color.copy(hovered.userData.base).offsetHSL(0, 0, 0.18);
      const l = hovered.userData.line;
      tooltip.innerHTML =
        `<b>${l.txn_id}</b> &middot; ${l.amount}<br>${l.value_date} &middot; ` +
        `${l.matched ? l.rows + " rows" : l.exception_type || "exception"}<br>` +
        `<span class="tip-reason">${l.reason}</span>`;
      tooltip.style.display = "block";
    }
    if (tooltip.style.display === "block") {
      const r = renderer.domElement.getBoundingClientRect();
      tooltip.style.left = Math.min(e.clientX - r.left + 14, r.width - 250) + "px";
      tooltip.style.top = e.clientY - r.top + 14 + "px";
    }
    renderer.domElement.style.cursor = hit ? "pointer" : "grab";
  });

  renderer.domElement.addEventListener("click", (e) => {
    const hit = pick(e);
    if (hit) onSelect(hit.object.userData.line.txn_id);
  });
  renderer.domElement.addEventListener("pointerleave", () => {
    tooltip.style.display = "none";
    if (hovered) {
      hovered.material.color.copy(hovered.userData.base);
      hovered = null;
    }
  });

  function resize() {
    const w = container.clientWidth || width;
    const h = Math.max(460, Math.min(780, Math.round(w * 0.74)));
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    const d = new THREE.Vector3(
      Math.sin(controls.state.phi) * Math.sin(controls.state.theta),
      Math.cos(controls.state.phi),
      Math.sin(controls.state.phi) * Math.cos(controls.state.theta)
    );
    controls.state.radius = fitDistance(bounds, camera, d);
    controls.apply();
  }
  window.addEventListener("resize", resize);

  const clock = new THREE.Clock();
  let raf = 0;
  function tick() {
    raf = requestAnimationFrame(tick);
    const t = clock.getElapsedTime();

    for (let i = animated.length - 1; i >= 0; i--) {
      const a = animated[i];
      const local = t - a.delay;
      if (local <= 0) continue;
      const k = Math.min(1, local / 0.85);
      const eased = 1 - Math.pow(1 - k, 3);
      a.mesh.position.y = a.restY + (1 - eased) * a.lift;
      if (k >= 1) animated.splice(i, 1);
    }

    if (idle) {
      controls.state.theta -= 0.0016;
      controls.apply();
    }
    renderer.render(scene, camera);
  }
  tick();

  return {
    dispose() {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      renderer.dispose();
      disposables.forEach((g) => g.dispose());
      container.innerHTML = "";
    },
  };
}
