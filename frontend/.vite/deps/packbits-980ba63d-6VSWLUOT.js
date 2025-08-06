import {
  xvi
} from "./chunk-BBHCMWC6.js";
import {
  require_react_dom
} from "./chunk-XXCRK5JG.js";
import {
  require_react
} from "./chunk-WNPTCGAH.js";
import {
  __toESM
} from "./chunk-5WRI5ZAA.js";

// node_modules/vitessce/dist/packbits-980ba63d.js
var import_react = __toESM(require_react(), 1);
var import_react_dom = __toESM(require_react_dom(), 1);
var p = class extends xvi {
  decodeBlock(n) {
    const r = new DataView(n), a = [];
    for (let e = 0; e < n.byteLength; ++e) {
      let t = r.getInt8(e);
      if (t < 0) {
        const o = r.getUint8(e + 1);
        t = -t;
        for (let s = 0; s <= t; ++s)
          a.push(o);
        e += 1;
      } else {
        for (let o = 0; o <= t; ++o)
          a.push(r.getUint8(e + o + 1));
        e += t + 1;
      }
    }
    return new Uint8Array(a).buffer;
  }
};
export {
  p as default
};
//# sourceMappingURL=packbits-980ba63d-6VSWLUOT.js.map
