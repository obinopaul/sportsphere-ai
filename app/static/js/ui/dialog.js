// app/static/js/ui/dialog.js
// Displays a modal dialog with a title, message, and a close button.
function showDialog(titleText, messageText) {
  // Create overlay
  const overlay = document.createElement('div');
  overlay.className = 'dialog-overlay';
  
  // Create dialog content
  const dialog = document.createElement('div');
  dialog.className = 'dialog-content';
  
  const title = document.createElement('h2');
  title.textContent = titleText;
  dialog.appendChild(title);
  
  const message = document.createElement('p');
  message.textContent = messageText;
  dialog.appendChild(message);
  
  const closeBtn = document.createElement('button');
  closeBtn.textContent = 'Close';
  closeBtn.className = 'btn btn-secondary';
  closeBtn.style.marginTop = '1rem';
  closeBtn.addEventListener('click', () => {
    document.body.removeChild(overlay);
  });
  dialog.appendChild(closeBtn);
  
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
}


// app/static/js/ui/dialog.js
// Creates a modal dialog element. Call openDialog() to display it.
function createDialog({ title = "", content = "", onClose = null }) {
  const overlay = document.createElement("div");
  overlay.className = "dialog-overlay";
  
  const dialog = document.createElement("div");
  dialog.className = "dialog-content";
  
  if (title) {
    const dialogTitle = document.createElement("h2");
    dialogTitle.className = "dialog-title";
    dialogTitle.textContent = title;
    dialog.appendChild(dialogTitle);
  }
  
  const dialogBody = document.createElement("div");
  dialogBody.className = "dialog-body";
  dialogBody.innerHTML = content;
  dialog.appendChild(dialogBody);
  
  const closeButton = createButton({
    variant: "secondary",
    text: "Close",
    ariaLabel: "Close dialog",
    onClick: () => {
      document.body.removeChild(overlay);
      if (typeof onClose === "function") {
        onClose();
      }
    }
  });
  closeButton.classList.add("dialog-close");
  dialog.appendChild(closeButton);
  
  overlay.appendChild(dialog);
  return overlay;
}
