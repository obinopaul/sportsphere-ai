// app/static/js/ui/number-ticker.js
// Creates a number ticker that animates from 0 (or a starting value) to the target value.
function createNumberTicker({ value = 0, duration = 2000, decimalPlaces = 0, className = "" }) {
  const ticker = document.createElement("span");
  ticker.className = "number-ticker " + className;
  let start = 0;
  const startTime = performance.now();
  
  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const currentValue = start + (value - start) * progress;
    ticker.textContent = currentValue.toFixed(decimalPlaces);
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  requestAnimationFrame(update);
  return ticker;
}
