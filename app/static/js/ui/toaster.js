// app/static/js/ui/toaster.js
// Creates a simple toaster for notifications.
function createToaster() {
    const toaster = document.createElement("div");
    toaster.className = "toaster";
    document.body.appendChild(toaster);
    
    function showToast({ message = "", duration = 3000 }) {
      const toast = document.createElement("div");
      toast.className = "toast";
      toast.textContent = message;
      toaster.appendChild(toast);
      setTimeout(() => {
        toaster.removeChild(toast);
      }, duration);
    }
    
    return {
      showToast
    };
  }
  