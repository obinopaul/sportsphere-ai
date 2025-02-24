// app/static/js/ui/card.js
// Creates a card element with optional header, title, description, content, and footer.
function createCard({ header = "", title = "", description = "", content = "", footer = "" }) {
  const card = document.createElement("div");
  card.className = "card";
  
  if (header) {
    const cardHeader = document.createElement("div");
    cardHeader.className = "card-header";
    cardHeader.innerHTML = header;
    card.appendChild(cardHeader);
  }
  
  if (title) {
    const cardTitle = document.createElement("h3");
    cardTitle.className = "card-title";
    cardTitle.textContent = title;
    card.appendChild(cardTitle);
  }
  
  if (description) {
    const cardDesc = document.createElement("p");
    cardDesc.className = "card-description";
    cardDesc.textContent = description;
    card.appendChild(cardDesc);
  }
  
  if (content) {
    const cardContent = document.createElement("div");
    cardContent.className = "card-content";
    cardContent.innerHTML = content;
    card.appendChild(cardContent);
  }
  
  if (footer) {
    const cardFooter = document.createElement("div");
    cardFooter.className = "card-footer";
    cardFooter.innerHTML = footer;
    card.appendChild(cardFooter);
  }
  
  return card;
}
