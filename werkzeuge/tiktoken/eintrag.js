// Einstieg für das Bündel docs/app/vendor/tiktoken.js.
// Bauen: npm --prefix werkzeuge/tiktoken install && npm --prefix werkzeuge/tiktoken run bauen

import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";
import cl100k_base from "js-tiktoken/ranks/cl100k_base";

const RANKS = { o200k_base, cl100k_base };
const fertig = {};
window.TIKTOKEN = {
  laden: name => (fertig[name] ??= new Tiktoken(RANKS[name])),
};
