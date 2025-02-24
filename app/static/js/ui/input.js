// app/static/js/ui/input.js
// Creates a styled input element.
function createInput({ type = "text", placeholder = "", className = "", onChange = null }) {
  const input = document.createElement("input");
  input.type = type;
  input.placeholder = placeholder;
  input.className = "input " + className;
  if (onChange && typeof onChange === "function") {
    input.addEventListener("input", onChange);
  }
  return input;
}
