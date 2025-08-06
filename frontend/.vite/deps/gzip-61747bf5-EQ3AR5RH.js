import {
  dr,
  pr
} from "./chunk-MUGINXH3.js";
import "./chunk-5WRI5ZAA.js";

// node_modules/vitessce/dist/gzip-61747bf5.js
var o = Object.defineProperty;
var a = (t, e, i) => e in t ? o(t, e, { enumerable: true, configurable: true, writable: true, value: i }) : t[e] = i;
var n = (t, e, i) => (a(t, typeof e != "symbol" ? e + "" : e, i), i);
var r;
var l = (r = class {
  constructor(e = 1) {
    n(this, "level");
    if (e < 0 || e > 9)
      throw new Error("Invalid gzip compression level, it should be between 0 and 9");
    this.level = e;
  }
  static fromConfig({ level: e }) {
    return new r(e);
  }
  encode(e) {
    return pr(e, { level: this.level });
  }
  decode(e) {
    return dr(e);
  }
}, n(r, "codecId", "gzip"), r);
var u = l;
export {
  u as default
};
//# sourceMappingURL=gzip-61747bf5-EQ3AR5RH.js.map
