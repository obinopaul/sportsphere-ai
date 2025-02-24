// app/static/js/ui/globe.js
// Creates an interactive globe. The container should be a div element.
function createGlobeComponent(container, config = {}) {
  const defaultConfig = {
    width: 800,
    height: 800,
    onRender: () => {},
    devicePixelRatio: 2,
    phi: 0,
    theta: 0.3,
    dark: 0,
    diffuse: 0.4,
    mapSamples: 16000,
    mapBrightness: 1.2,
    baseColor: [1, 1, 1],
    markerColor: [251 / 255, 100 / 255, 21 / 255],
    glowColor: [1, 1, 1],
    markers: [
      { location: [14.5995, 120.9842], size: 0.03 },
      { location: [19.076, 72.8777], size: 0.1 },
      // Add other markers as desired
    ]
  };
  
  const finalConfig = Object.assign({}, defaultConfig, config);
  let phi = 0;
  let width = container.offsetWidth;
  
  function onRender(state) {
    phi += 0.005;
    state.phi = phi;
    state.width = width * 2;
    state.height = width * 2;
    finalConfig.onRender(state);
  }
  
  // Call the globe library's function (assumed to be globally available)
  const globeInstance = createGlobe(container, Object.assign({}, finalConfig, {
    width: width * 2,
    height: width * 2,
    onRender: onRender
  }));
  
  window.addEventListener("resize", () => {
    width = container.offsetWidth;
  });
  
  return globeInstance;
}
