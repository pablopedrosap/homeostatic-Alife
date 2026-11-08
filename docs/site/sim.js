// Faithful JS port of homeostatic_colony/alife/emergent_world.py (r-regime defaults,
// HAB-E2v weather + yoked control). Same constants, same update order. Different RNG
// stream, so trajectories match the Python statistically, not bit for bit.
(function (root) {
  const CFG = {
    W: 30, H: 30, nN: 10, nT: 5, nC: 4, rN: 3.2, rT: 3.2, rC: 2.6,
    basal: 0.004, moveCost: 0.010, nutGain: 0.13, heatSpeed: 0.09, heatHazard: 0.05,
    cooling: 0.016, coolZone: 0.03, dmgToxin: 0.035, dmgOverheat: 0.05, repair: 0.010,
    repairMinE: 0.30, tCrit: 0.7, tFatal: 1.2, dMax: 1.0, maxSpeed: 0.35,
    eInit: 0.6, tInit: 0.3, cap: 500, nInit: 150, divE: 0.78, divCost: 0.30, minAge: 25,
    nH: 8, wStd: 0.5, mut: 0.08, clip: 3.0, eta: 0.02, tau: 50,
  };
  const NI = 8, NO = 2, NH = CFG.nH;

  function rngFrom(seed) {
    let a = seed >>> 0;
    const u = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    const n = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x, y, r;
      do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
      const f = Math.sqrt(-2 * Math.log(r) / r);
      spare = y * f; return x * f;
    };
    return { u, n };
  }

  class World {
    constructor(opts = {}) {
      this.c = Object.assign({}, CFG);
      this.k = opts.k ?? 0;            // weather amplitude in units of cooling rate
      this.senseFrac = opts.senseFrac ?? 0.5;
      this.reset(opts.seed ?? 1);
    }
    reset(seed) {
      const c = this.c, C = c.cap;
      this.seed = seed;
      this.rng = rngFrom(seed);
      const R = this.rng;
      const place = (n, r) => Array.from({ length: n }, () => [r + R.u() * (c.W - 2 * r), r + R.u() * (c.H - 2 * r)]);
      this.nut = place(c.nN, c.rN); this.tox = place(c.nT, c.rT); this.cool = place(c.nC, c.rC);
      this.x = new Float32Array(C); this.y = new Float32Array(C);
      this.E = new Float32Array(C); this.T = new Float32Array(C); this.D = new Float32Array(C);
      this.age = new Int32Array(C); this.gen = new Int32Array(C);
      this.alive = new Uint8Array(C); this.senses = new Uint8Array(C); this.partner = new Int32Array(C);
      this.gW1 = new Float32Array(C * NH * NI); this.gW2 = new Float32Array(C * NO * NH);
      this.b1 = new Float32Array(C * NH); this.b2 = new Float32Array(C * NO);
      this.W1 = new Float32Array(C * NH * NI); this.W2 = new Float32Array(C * NO * NH);
      this.speed = new Float32Array(C);
      this.t = 0; this.weather = 0; this.births = 0; this.deaths = 0;
      this.heatDeaths = [0, 0];
      const nS = Math.round(this.senseFrac * c.nInit);
      const order = Array.from({ length: c.nInit }, (_, i) => i);
      for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(R.u() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      const sensing = new Uint8Array(c.nInit); for (let i = 0; i < nS; i++) sensing[order[i]] = 1;
      for (let i = 0; i < c.nInit; i++) {
        for (let j = 0; j < NH * NI; j++) this.gW1[i * NH * NI + j] = R.n() * c.wStd;
        for (let j = 0; j < NO * NH; j++) this.gW2[i * NO * NH + j] = R.n() * c.wStd;
        for (let j = 0; j < NH; j++) this.b1[i * NH + j] = R.n() * 0.1;
        for (let j = 0; j < NO; j++) this.b2[i * NO + j] = R.n() * 0.1;
        this._spawn(i, 0, sensing[i]);
        this.x[i] = 1 + R.u() * (c.W - 2); this.y[i] = 1 + R.u() * (c.H - 2);
      }
    }
    _spawn(i, gen, senses) {
      const c = this.c;
      this.W1.set(this.gW1.subarray(i * NH * NI, (i + 1) * NH * NI), i * NH * NI);
      this.W2.set(this.gW2.subarray(i * NO * NH, (i + 1) * NO * NH), i * NO * NH);
      this.E[i] = c.eInit; this.T[i] = c.tInit; this.D[i] = 0; this.age[i] = 0;
      this.gen[i] = gen; this.alive[i] = 1; this.senses[i] = senses; this.partner[i] = i;
    }
    static field(px, py, src, r) {
      let s = 0; const k = -0.5 / (r * r);
      for (const [cx, cy] of src) { const dx = px - cx, dy = py - cy; s += Math.exp(k * (dx * dx + dy * dy)); }
      return s;
    }
    sample(px, py) {
      const c = this.c, F = World.field;
      return [F(px, py, this.nut, c.rN), F(px, py, this.tox, c.rT), F(px, py, this.cool, c.rC)];
    }
    step() {
      const c = this.c, R = this.rng, C = c.cap, F = World.field;
      this.t++;
      let heatLoad = 0;
      if (this.k > 0) {
        const rho = Math.exp(-1 / c.tau);
        this.weather = rho * this.weather + Math.sqrt(1 - rho * rho) * R.n();
        heatLoad = this.k * c.cooling * this.weather;
      }
      const living = []; for (let i = 0; i < C; i++) if (this.alive[i]) living.push(i);
      if (!living.length) return;
      // yoked partners
      for (const i of living) {
        if (!this.senses[i]) {
          const p = this.partner[i];
          if (p === i || !this.alive[p]) {
            if (living.length > 1) { let q = living[Math.floor(R.u() * living.length)]; if (q === i) q = living[(living.indexOf(q) + 1) % living.length]; this.partner[i] = q; }
          }
        }
      }
      const eps = 0.3, obs = new Float32Array(NI), h = new Float32Array(NH), mv = new Float32Array(NO);
      const Es = this.E.slice(), Ts = this.T.slice(), Ds = this.D.slice();
      for (const i of living) {
        const px = this.x[i], py = this.y[i];
        obs[0] = Math.max(-1, Math.min(1, (F(px + eps, py, this.nut, c.rN) - F(px - eps, py, this.nut, c.rN)) / (2 * eps)));
        obs[1] = Math.max(-1, Math.min(1, (F(px, py + eps, this.nut, c.rN) - F(px, py - eps, this.nut, c.rN)) / (2 * eps)));
        obs[2] = Math.max(-1, Math.min(1, (F(px + eps, py, this.tox, c.rT) - F(px - eps, py, this.tox, c.rT)) / (2 * eps)));
        obs[3] = Math.max(-1, Math.min(1, (F(px, py + eps, this.tox, c.rT) - F(px, py - eps, this.tox, c.rT)) / (2 * eps)));
        obs[4] = F(px, py, this.cool, c.rC);
        const src = this.senses[i] ? i : this.partner[i];
        obs[5] = Es[src]; obs[6] = Ts[src]; obs[7] = Ds[src];
        const o1 = i * NH * NI, o2 = i * NO * NH;
        for (let a = 0; a < NH; a++) { let s = this.b1[i * NH + a]; for (let b = 0; b < NI; b++) s += this.W1[o1 + a * NI + b] * obs[b]; h[a] = Math.tanh(s); }
        for (let a = 0; a < NO; a++) { let s = this.b2[i * NO + a]; for (let b = 0; b < NH; b++) s += this.W2[o2 + a * NH + b] * h[b]; mv[a] = Math.tanh(s); }
        // Oja plasticity (reward-free)
        for (let a = 0; a < NH; a++) for (let b = 0; b < NI; b++) {
          const j = o1 + a * NI + b; let w = this.W1[j] + c.eta * (h[a] * obs[b] - h[a] * h[a] * this.W1[j]);
          this.W1[j] = Math.max(-c.clip, Math.min(c.clip, w));
        }
        for (let a = 0; a < NO; a++) for (let b = 0; b < NH; b++) {
          const j = o2 + a * NH + b; let w = this.W2[j] + c.eta * (mv[a] * h[b] - mv[a] * mv[a] * this.W2[j]);
          this.W2[j] = Math.max(-c.clip, Math.min(c.clip, w));
        }
        const sp = Math.hypot(mv[0], mv[1]);
        const sc = Math.min(1, c.maxSpeed / Math.max(sp, 1e-6));
        const mx = mv[0] * sc, my = mv[1] * sc, mag = Math.hypot(mx, my);
        this.speed[i] = mag;
        this.x[i] = Math.max(0, Math.min(c.W, px + mx)); this.y[i] = Math.max(0, Math.min(c.H, py + my));
        this.age[i]++;
        const [nu, tx, co] = this.sample(this.x[i], this.y[i]);
        this.E[i] = Math.max(0, Math.min(1, this.E[i] - c.basal - c.moveCost * mag + c.nutGain * nu));
        this.T[i] = Math.max(0, Math.min(1.5, this.T[i] + c.heatSpeed * mag + c.heatHazard * tx - (c.cooling + c.coolZone * co) + heatLoad));
        const over = Math.max(0, this.T[i] - c.tCrit);
        const rest = mag < 0.4 * c.maxSpeed ? 1 : 0;
        const rep = c.repair * rest * (this.E[i] > c.repairMinE ? 1 : 0.35);
        this.D[i] = Math.max(0, Math.min(c.dMax, this.D[i] + c.dmgToxin * tx + c.dmgOverheat * over - rep));
      }
      for (const i of living) {
        if (this.E[i] <= 0 || this.T[i] >= c.tFatal || this.D[i] >= c.dMax) {
          if (this.T[i] >= c.tFatal) this.heatDeaths[this.senses[i] ? 0 : 1]++;
          this.alive[i] = 0; this.deaths++;
        }
      }
      let pop = 0; for (let i = 0; i < C; i++) pop += this.alive[i];
      for (const p of living) {
        if (!this.alive[p] || this.E[p] <= c.divE || this.age[p] < c.minAge) continue;
        if (pop >= C) break;
        let ci = -1; for (let j = 0; j < C; j++) if (!this.alive[j]) { ci = j; break; }
        if (ci < 0) break;
        const avail = this.E[p] - c.divCost;
        if (avail <= 0.2) continue;
        const cl = (v) => Math.max(-c.clip, Math.min(c.clip, v));
        for (let j = 0; j < NH * NI; j++) this.gW1[ci * NH * NI + j] = cl(this.gW1[p * NH * NI + j] + R.n() * c.mut);
        for (let j = 0; j < NO * NH; j++) this.gW2[ci * NO * NH + j] = cl(this.gW2[p * NO * NH + j] + R.n() * c.mut);
        for (let j = 0; j < NH; j++) this.b1[ci * NH + j] = this.b1[p * NH + j] + R.n() * c.mut;
        for (let j = 0; j < NO; j++) this.b2[ci * NO + j] = this.b2[p * NO + j] + R.n() * c.mut;
        this._spawn(ci, this.gen[p] + 1, this.senses[p]);
        this.E[p] = avail * 0.5; this.E[ci] = avail * 0.5;
        this.x[ci] = Math.max(0, Math.min(c.W, this.x[p] + R.n() * 0.5));
        this.y[ci] = Math.max(0, Math.min(c.H, this.y[p] + R.n() * 0.5));
        this.births++; pop++;
      }
    }
    stats() {
      let pop = 0, nS = 0, tS = 0, tY = 0, mg = 0;
      for (let i = 0; i < this.c.cap; i++) if (this.alive[i]) {
        pop++; if (this.senses[i]) { nS++; tS += this.T[i]; } else tY += this.T[i];
        if (this.gen[i] > mg) mg = this.gen[i];
      }
      return { t: this.t, pop, nS, pS: pop ? nS / pop : 0, TS: nS ? tS / nS : null,
               TY: pop - nS ? tY / (pop - nS) : null, maxGen: mg, weather: this.weather };
    }
  }
  root.HomeoWorld = World;
  if (typeof module !== "undefined") module.exports = World;
})(typeof window !== "undefined" ? window : globalThis);
