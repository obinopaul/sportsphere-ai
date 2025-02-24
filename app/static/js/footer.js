// app/static/js/footer.js
// Renders an enlarged footer with larger social media icons and attribution text.

(function() {
  const footerEl = document.getElementById('site-footer');
  if (!footerEl) return;

  // Set container styles to make the footer larger.
  footerEl.style.width = "100%";
  footerEl.style.display = "flex";
  footerEl.style.flexDirection = "column";
  footerEl.style.alignItems = "center";
  footerEl.style.justifyContent = "center";
  footerEl.style.padding = "2rem"; // Increased overall padding.
  footerEl.style.borderTop = "1px solid #e5e7eb"; // Light top border.
  footerEl.style.backgroundColor = "#f9f9f9"; // Optional background color.

  // Attribution text with gradient styling and larger font.
  const attribution = document.createElement('p');
  attribution.textContent = 'Made by Paul Okafor';
  attribution.style.fontWeight = "600";
  attribution.style.fontSize = "1rem"; // Larger font size.
  attribution.style.background = "linear-gradient(45deg, rgba(59,130,246,0.9), rgba(37,99,235,0.9))";
  attribution.style.webkitBackgroundClip = "text";
  attribution.style.webkitTextFillColor = "transparent";
  attribution.style.marginBottom = "1rem";
  footerEl.appendChild(attribution);

  // Social icons array using provided image URLs.
  const socialIcons = [
    {
      name: "LinkedIn",
      imgSrc: "https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg",
      link: "https://linkedin.com/in/obinopaul"
    },
    {
      name: "Github",
      imgSrc: "https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/github.svg",
      link: "https://github.com/obinopaul"
    },
    {
      name: "Facebook",
      imgSrc: "https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/facebook.svg",
      link: "https://fb.com/paultwizzy"
    },
    {
      name: "Instagram",
      imgSrc: "https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg",
      link: "https://www.instagram.com/obinopaul/"
    },
    {
      name: "Personal Website",
      imgSrc: "https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/codepen.svg",
      link: "https://obinopaul.com/"
    }
  ];

  // Create a container for the social icons.
  const socialContainer = document.createElement('div');
  socialContainer.style.display = "flex";
  socialContainer.style.alignItems = "center";
  socialContainer.style.justifyContent = "center";
  socialContainer.style.gap = "1rem";

  // Create each social link with the provided image icons.
  socialIcons.forEach(social => {
    const a = document.createElement('a');
    a.href = social.link;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.title = social.name;
    a.style.display = "flex";
    a.style.alignItems = "center";
    a.style.justifyContent = "center";
    a.style.textDecoration = "none";

    // Create the image element for the icon.
    const img = document.createElement('img');
    img.src = social.imgSrc;
    img.alt = social.name;
    // Increase the size of the icons.
    img.style.height = "30px";  // Larger icon height.
    img.style.width = "30px";   // Larger icon width.
    img.style.objectFit = "contain";
    img.style.transition = "transform 0.3s";
    // Add a hover effect.
    a.addEventListener("mouseover", () => {
      img.style.transform = "scale(1.1)";
    });
    a.addEventListener("mouseout", () => {
      img.style.transform = "scale(1)";
    });

    a.appendChild(img);
    socialContainer.appendChild(a);
  });

  footerEl.appendChild(socialContainer);


  // --------------------------------------- Responsive Adjustments ---------------------------------------
  // Responsive Adjustments
  function adjustFooter() {
    if (window.innerWidth <= 768) {
      footerEl.style.flexDirection = 'column';
      footerEl.style.textAlign = 'center';
      socialContainer.style.flexDirection = 'column';
      socialContainer.style.gap = '0.5rem';
    } else {
      footerEl.style.flexDirection = 'row';
      socialContainer.style.flexDirection = 'row';
    }
  }

  // Run on page load
  adjustFooter();

  // Listen for window resize
  window.addEventListener('resize', adjustFooter);

  // -------------------------------------------------------------------------------------------------------------
})();
