<template>
  <router-view />
  <!--
    AGPL-3.0 §13 compliance: when this software is used over a network,
    we must make the corresponding source available to its users. The
    link below points at the public fork that is actually deployed.
  -->
  <footer class="agpl-footer">
    <a
      :href="sourceUrl"
      target="_blank"
      rel="noopener noreferrer"
    >Source code</a>
    <span class="agpl-sep">·</span>
    <a
      href="https://www.gnu.org/licenses/agpl-3.0.html"
      target="_blank"
      rel="noopener noreferrer"
    >AGPL-3.0</a>
  </footer>
</template>

<script setup>
// VITE_SOURCE_URL can be injected at build time via --build-arg in the
// Dockerfile (e.g. https://github.com/<user>/mirofish101/tree/<sha>).
// Falls back to the public fork root if unset so the link is never broken.
const sourceUrl =
  import.meta.env.VITE_SOURCE_URL ||
  'https://github.com/nnfuzzy/mirofish101';
</script>

<style>
/* 全局样式重置 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

#app {
  font-family: 'JetBrains Mono', 'Space Grotesk', 'Noto Sans SC', monospace;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #000000;
  background-color: #ffffff;
}

/* 滚动条样式 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
  background: #000000;
}

::-webkit-scrollbar-thumb:hover {
  background: #333333;
}

/* 全局按钮样式 */
button {
  font-family: inherit;
}

/* AGPL footer — small, fixed to bottom-right, low visual weight. */
.agpl-footer {
  position: fixed;
  bottom: 6px;
  right: 10px;
  z-index: 9999;
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  color: rgba(0, 0, 0, 0.55);
  background: rgba(255, 255, 255, 0.85);
  padding: 2px 8px;
  border-radius: 3px;
  pointer-events: auto;
}

.agpl-footer a {
  color: inherit;
  text-decoration: underline;
  text-decoration-color: rgba(0, 0, 0, 0.3);
}

.agpl-footer a:hover {
  color: #000;
  text-decoration-color: #000;
}

.agpl-sep {
  margin: 0 6px;
  opacity: 0.5;
}
</style>
