import {
  aa,
  ra
} from "./chunk-MUGINXH3.js";
import "./chunk-5WRI5ZAA.js";

// node_modules/vitessce/dist/zlib-99872a4e.js
var n = Object.defineProperty;
var o = (l, e, t) => e in l ? n(l, e, { enumerable: true, configurable: true, writable: true, value: t }) : l[e] = t;
var i = (l, e, t) => (o(l, typeof e != "symbol" ? e + "" : e, t), t);
var r;
var c = (r = class {
  constructor(e = 1) {
    i(this, "level");
    if (e < -1 || e > 9)
      throw new Error("Invalid zlib compression level, it should be between -1 and 9");
    this.level = e;
  }
  static fromConfig({ level: e }) {
    return new r(e);
  }
  encode(e) {
    return ra(e, { level: this.level });
  }
  decode(e) {
    return aa(e);
  }
}, i(r, "codecId", "zlib"), r);
var u = c;
export {
  u as default
};
//# sourceMappingURL=zlib-99872a4e-3KBUSICA.js.map
