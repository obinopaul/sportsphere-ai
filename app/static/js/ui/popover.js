// app/static/js/ui/popover.js
// Creates a popover element that toggles visibility when the trigger is clicked.
function createPopover({ triggerText = "Open", contentHTML = "", className = "" }) {
  const container = document.createElement("div");
  container.className = "popover-container " + className;
  
  const trigger = createButton({
    variant: "default",
    text: triggerText,
    ariaLabel: "Toggle popover"
  });
  trigger.classList.add("popover-trigger");
  container.appendChild(trigger);
  
  const popoverContent = document.createElement("div");
  popoverContent.className = "popover-content";
  popoverContent.innerHTML = contentHTML;
  container.appendChild(popoverContent);
  
  popoverContent.style.display = "none";
  
  trigger.addEventListener("click", () => {
    popoverContent.style.display = popoverContent.style.display === "none" ? "block" : "none";
  });
  
  document.addEventListener("click", (e) => {
    if (!container.contains(e.target)) {
      popoverContent.style.display = "none";
    }
  });
  
  return container;
}
