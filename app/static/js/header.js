// app/static/js/header.js
// Renders the header with a clickable logo and title (both linking to home)
// and a theme toggle button that displays an icon.

(function() {
  const headerEl = document.getElementById('site-header');
  if (!headerEl) return;

  // Create a container for the left side (logo + title)
  const leftContainer = document.createElement('div');
  leftContainer.style.display = 'flex';
  leftContainer.style.alignItems = 'center';

  // Create a link that wraps the logo
  const logoLink = document.createElement('a');
  logoLink.href = '/';
  // Create the logo image element
  const logo = document.createElement('img');
  logo.src = '/static/images/logo.png'; // Ensure your logo is placed in /static/images/
  logo.alt = 'PocketTraveller Logo';
  logo.style.height = '40px';
  logo.style.marginRight = '0.5rem';
  // Append the logo to its link
  logoLink.appendChild(logo);

  // Create a link that wraps the title text
  const titleLink = document.createElement('a');
  titleLink.href = '/';
  // Create the title element with a gradient effect and custom font
  const title = document.createElement('h1');
  title.textContent = 'PocketTraveller';
  title.style.fontFamily = 'Satoshi-Bold'; // Using one of your Satoshi font families
  title.style.fontSize = '1.8rem';
  title.style.background = 'linear-gradient(45deg, #3b82f6, #2563eb)';
  title.style.webkitBackgroundClip = 'text';
  title.style.webkitTextFillColor = 'transparent';
  // Remove any default link styling
  titleLink.style.textDecoration = 'none';
  titleLink.appendChild(title);

  // Append both the logo link and title link to the left container
  leftContainer.appendChild(logoLink);
  leftContainer.appendChild(titleLink);

  // Create the theme toggle button with an icon (using Font Awesome icon here)
  const toggleBtn = document.createElement('button');
  toggleBtn.className = 'btn btn-secondary';
  // You can use a Unicode icon (🌙) or Font Awesome (if included)
  // toggleBtn.innerHTML = '<i class="fas fa-adjust"></i>';
  toggleBtn.textContent = 'Toggle Theme';
  // Push the toggle button to the far right
  toggleBtn.style.marginLeft = 'auto';
  toggleBtn.addEventListener('click', () => {
    // Toggle dark/light mode by toggling a CSS class on the body
    document.body.classList.toggle('dark-mode');
  });

  // Set header container styles and append the left container and toggle button
  headerEl.style.display = 'flex';
  headerEl.style.alignItems = 'center';
  headerEl.style.justifyContent = 'space-between';
  headerEl.style.padding = '1rem';
  headerEl.appendChild(leftContainer);
  headerEl.appendChild(toggleBtn);


  // ----------------------------- Responsive Adjustments ----------------------------- //
  // Responsive Adjustments
  function adjustHeader() {
    if (window.innerWidth <= 768) {
      headerEl.style.flexDirection = 'column';
      headerEl.style.textAlign = 'center';
      leftContainer.style.justifyContent = 'center';
      leftContainer.style.flexDirection = 'column';
      toggleBtn.style.margin = '1rem 0';
    } else {
      headerEl.style.flexDirection = 'row';
      leftContainer.style.flexDirection = 'row';
      toggleBtn.style.marginLeft = 'auto';
    }
  }

  // Run on page load
  adjustHeader();

  // Listen for window resize
  window.addEventListener('resize', adjustHeader);
  // ---------------------------------------------------------------------------------------------------
})();
