// app/static/js/ui/button.js
// Creates a button element with variant and size options.
function createButton({ variant = "default", size = "default", text = "", onClick = null, ariaLabel = "" }) {
  const btn = document.createElement("button");
  btn.setAttribute("aria-label", ariaLabel);
  btn.textContent = text;
  
  // Add variant classes
  let classes = "btn ";
  switch(variant) {
    case "default":
      classes += "btn-default ";
      break;
    case "destructive":
      classes += "btn-destructive ";
      break;
    case "outline":
      classes += "btn-outline ";
      break;
    case "secondary":
      classes += "btn-secondary ";
      break;
    case "ghost":
      classes += "btn-ghost ";
      break;
    case "link":
      classes += "btn-link ";
      break;
    default:
      classes += "btn-default ";
      break;
  }
  // Add size classes
  switch(size) {
    case "default":
      classes += "btn-size-default ";
      break;
    case "sm":
      classes += "btn-size-sm ";
      break;
    case "lg":
      classes += "btn-size-lg ";
      break;
    case "icon":
      classes += "btn-size-icon ";
      break;
    default:
      classes += "btn-size-default ";
      break;
  }
  btn.className = classes.trim();
  if (onClick && typeof onClick === "function") {
    btn.addEventListener("click", onClick);
  }
  return btn;
}
