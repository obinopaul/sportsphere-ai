// app/static/js/ui/dropdown-menu.js
// Creates a simple dropdown menu.
function createDropdownMenu({ triggerText = "Menu", items = [] }) {
  const container = document.createElement("div");
  container.className = "dropdown-menu-container";
  
  const trigger = createButton({
    variant: "default",
    text: triggerText,
    ariaLabel: "Open menu"
  });
  trigger.classList.add("dropdown-trigger");
  container.appendChild(trigger);
  
  const menu = document.createElement("ul");
  menu.className = "dropdown-menu";
  items.forEach(item => {
    const li = document.createElement("li");
    li.className = "dropdown-menu-item";
    const a = document.createElement("a");
    a.href = item.link || "#";
    a.textContent = item.label;
    li.appendChild(a);
    menu.appendChild(li);
  });
  container.appendChild(menu);
  
  // Toggle the menu visibility when clicking the trigger
  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    menu.classList.toggle("show");
  });
  
  // Hide the menu when clicking outside
  document.addEventListener("click", function(e) {
    if (!container.contains(e.target)) {
      menu.classList.remove("show");
    }
  });
  
  return container;
}
