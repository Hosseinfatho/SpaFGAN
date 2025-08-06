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

// node_modules/vitessce/dist/webimage-5dc5b0d9.js
var import_react = __toESM(require_react(), 1);
var import_react_dom = __toESM(require_react_dom(), 1);
var m = class extends xvi {
  constructor() {
    if (super(), typeof createImageBitmap > "u")
      throw new Error("Cannot decode WebImage as `createImageBitmap` is not available");
    if (typeof document > "u" && typeof OffscreenCanvas > "u")
      throw new Error("Cannot decode WebImage as neither `document` nor `OffscreenCanvas` is not available");
  }
  async decode(d, n) {
    const o = new Blob([n]), e = await createImageBitmap(o);
    let t;
    typeof document < "u" ? (t = document.createElement("canvas"), t.width = e.width, t.height = e.height) : t = new OffscreenCanvas(e.width, e.height);
    const a = t.getContext("2d");
    return a.drawImage(e, 0, 0), a.getImageData(0, 0, e.width, e.height).data.buffer;
  }
};
export {
  m as default
};
//# sourceMappingURL=webimage-5dc5b0d9-F4SW7S3M.js.map
