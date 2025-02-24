// app/static/js/index.js
// Renders the hero section with a bold heading, subheading, buttons, and a scrolling marquee.

// Data for images in the marquee (adjust paths and links as needed)
const images = [
  {
    name: "Chichen Itza",
    src: "/static/js/hero/chichen.webp",
    link: "https://en.wikipedia.org/wiki/Chichen_Itza",
  },
  {
    name: "Christ the Redeemer",
    src: "/static/js/hero/christ.webp",
    link: "https://en.wikipedia.org/wiki/Christ_the_Redeemer_(statue)",
  },
  {
    name: "Colosseum",
    src: "/static/js/hero/colosseum.webp",
    link: "https://en.wikipedia.org/wiki/Colosseum",
  },
  {
    name: "Great Pyramid of Giza",
    src: "/static/js/hero/giza.webp",
    link: "https://en.wikipedia.org/wiki/Great_Pyramid_of_Giza",
  },
  {
    name: "Machu Picchu",
    src: "/static/js/hero/peru.webp",
    link: "https://en.wikipedia.org/wiki/Machu_Picchu",
  },
  {
    name: "Taj Mahal",
    src: "/static/js/hero/taj.webp",
    link: "https://en.wikipedia.org/wiki/Taj_Mahal",
  },
  {
    name: "India Gate",
    src: "/static/js/hero/india.webp",
    link: "https://en.wikipedia.org/wiki/India_Gate",
  },
  {
    name: "Great Wall of China",
    src: "/static/js/hero/wall.webp",
    link: "https://en.wikipedia.org/wiki/Great_Wall_of_China",
  },
  {
    name: "Eiffel Tower",
    src: "/static/js/hero/tower.webp",
    link: "https://en.wikipedia.org/wiki/Eiffel_Tower",
  },
  {
    name: "Statue of Liberty",
    src: "/static/js/hero/liberty.webp",
    link: "https://en.wikipedia.org/wiki/Statue_of_Liberty",
  },
  {
    name: "Sydney Opera House",
    src: "/static/js/hero/sydney.webp",
    link: "https://en.wikipedia.org/wiki/Sydney_Opera_House",
  },
  {
    name: "Mount Everest",
    src: "/static/js/hero/everest.webp",
    link: "https://en.wikipedia.org/wiki/Mount_Everest",
  },
  {
    name: "Stonehenge",
    src: "/static/js/hero/stonehenge.webp",
    link: "https://en.wikipedia.org/wiki/Stonehenge",
  },
];

// Split images into two rows
const half = Math.ceil(images.length / 2);
const firstRow = images.slice(0, half);
const secondRow = images.slice(half);


// Build the hero section
(function renderHero() {
  const heroEl = document.getElementById('hero-section');
  if (!heroEl) return;

  // Set hero section styling to fill the viewport and center content.
  heroEl.style.display = "flex";
  heroEl.style.flexDirection = "column";
  heroEl.style.alignItems = "center";
  heroEl.style.justifyContent = "center";
  heroEl.style.textAlign = "center";
  heroEl.style.minHeight = "100vh";
  heroEl.style.padding = "2rem 0";

  // Create a text container
  const textContainer = document.createElement("div");
  textContainer.style.padding = "1rem 2rem";
  textContainer.style.display = "flex";
  textContainer.style.flexDirection = "column";
  textContainer.style.alignItems = "center";
  textContainer.style.justifyContent = "center";
  textContainer.style.gap = "1rem";
  textContainer.style.maxWidth = "1200px";

  // --- Heading Section ---

  // // First line: "Embark on Electrifying Adventures with"
  // const line1 = document.createElement("h1");
  // line1.textContent = "Embark on Electrifying";
  // line1.style.fontWeight = "900";
  // line1.style.fontSize = "2.5rem"; // Larger than before
  // line1.style.lineHeight = "1.2";
  // // Use a warm gradient: coral to orange-red
  // line1.style.background = "linear-gradient(to bottom, rgba(59,130,246,0.9), rgba(37,99,235,0.6))";
  // line1.style.backgroundClip = "text";
  // line1.style.webkitBackgroundClip = "text";
  // line1.style.color = "transparent";
  // textContainer.appendChild(line1);

  // // First line: "Embark on Electrifying Adventures with"
  // const line1x = document.createElement("h1");
  // line1x.textContent = "Adventures with";
  // line1x.style.fontWeight = "900";
  // line1x.style.fontSize = "2.5rem"; // Larger than before
  // line1x.style.lineHeight = "1.2";
  // // Use a warm gradient: coral to orange-red
  // line1x.style.background = "linear-gradient(to bottom, rgba(59,130,246,0.9), rgba(37,99,235,0.6))";
  // line1x.style.backgroundClip = "text";
  // line1x.style.webkitBackgroundClip = "text";
  // line1x.style.color = "transparent";
  // textContainer.appendChild(line1x);

  // Import Google Fonts in your CSS file or HTML head:
  const fontLink = document.createElement("link");
  fontLink.rel = "stylesheet";
  fontLink.href = "https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Roboto:wght@400&display=swap";
  document.head.appendChild(fontLink);

  // First line: "Embark on Electrifying"
  const line1 = document.createElement("h1");
  line1.textContent = "Embark on Electrifying";
  line1.style.fontFamily = '"Playfair Display", Playfair Display';
  line1.style.fontWeight = "500";
  line1.style.fontSize = "3.5rem";
  line1.style.lineHeight = "0.2";
  // Use a custom CSS variable for the hero text color
  line1.style.color = "var(--hero-text-color)";
  textContainer.appendChild(line1);

  // Second line: "Adventures with"
  const line1x = document.createElement("h1");
  line1x.textContent = "Adventures with";
  line1x.style.fontFamily =  '"Playfair Display", Playfair Display';
  line1x.style.fontWeight = "500";
  line1x.style.fontSize = "3.5rem";
  line1x.style.lineHeight = "1.2";
  // Use the same custom CSS variable so it toggles with the theme
  line1x.style.color = "var(--hero-text-color)";
  textContainer.appendChild(line1x);


  // Second line: "Pocket Traveller"
  const line2 = document.createElement("h1");
  line2.textContent = "PocketTraveller";
  line2.style.fontFamily = '"Montserrat", sans-serif';
  line2.style.fontWeight = "900";
  line2.style.fontSize = "7rem"; // Even larger
  line2.style.lineHeight = "1.1";
  // Use a vibrant green gradient: lime green to forest green
  line2.style.background = "linear-gradient(45deg,  rgba(59,130,246,0.9), rgba(37,99,235,0.6))";
  line2.style.backgroundClip = "text";
  line2.style.webkitBackgroundClip = "text";
  line2.style.color = "transparent";
  line2.style.paddingBottom = "1rem";
  textContainer.appendChild(line2);

  // Third line: "Your trusted trip planner and adventure guide."
  const line3 = document.createElement("p");
  line3.textContent = "Your trusted AI travel agent and adventure guide.";
  line3.style.fontSize = "1.5rem"; // Slightly larger subheading
  line3.style.fontWeight = "500";
  line3.style.letterSpacing = "0.05em"; // tracking-tight
  // Use a gold-to-orange gradient
  // line3.style.background = "linear-gradient(45deg, #FFD700, #FFA500)";
  line3.style.color = "var(--hero-text-color)";
  line3.style.backgroundClip = "text";
  line3.style.webkitBackgroundClip = "text";
  // line3.style.color = "transparent";
  textContainer.appendChild(line3);

  // --- Buttons Section ---
  const btnContainer = document.createElement("div");
  btnContainer.style.display = "flex";
  btnContainer.style.flexDirection = "row";
  btnContainer.style.gap = "1rem";
  btnContainer.style.marginTop = "1rem";

  // "Get Started" button (links to trip planner)
  const getStartedBtn = document.createElement("a");
  getStartedBtn.href = "/plan-a-trip";
  getStartedBtn.textContent = "Let's Plan a Trip";
  // getStartedBtn.className = "btn btn-primary"; // ensure this class is styled in your CSS
  getStartedBtn.style.display = "inline-block";
  getStartedBtn.style.padding = "1rem 2rem";
  getStartedBtn.style.borderRadius = "0.7rem";
  getStartedBtn.style.textDecoration = "none";
  // getStartedBtn.style.fontWeight = "bold";
  getStartedBtn.style.fontSize = "1rem";
  // Use CSS variables for colors so that theme toggling works automatically.
  getStartedBtn.style.backgroundColor = "var(--btn-primary-bg)";
  getStartedBtn.style.color = "var(--btn-primary-text)";
  getStartedBtn.className = "btn btn-primary";
  btnContainer.appendChild(getStartedBtn);

  // "Buy Me a Coffee" button (external link)
  const coffeeBtn = document.createElement("a");
  coffeeBtn.href = "https://buymeacoffee.com/acobapaulf";
  coffeeBtn.target = "_blank";
  coffeeBtn.textContent = "Buy Me a Coffee";
  // coffeeBtn.className = "btn btn-secondary"; // ensure this class is styled in your CSS
  coffeeBtn.style.display = "inline-block";
  coffeeBtn.style.padding = "1rem 2rem";
  coffeeBtn.style.borderRadius = "0.7rem";
  coffeeBtn.style.textDecoration = "none";
  // coffeeBtn.style.fontWeight = "bold";
  coffeeBtn.style.fontSize = "1rem";
  // Use CSS variables for colors.
  coffeeBtn.style.backgroundColor = "var(--btn-secondary-bg)";
  coffeeBtn.style.color = "var(--btn-secondary-text)";
  coffeeBtn.style.border = "1px solid var(--btn-secondary-text)";
  coffeeBtn.className = "btn btn-secondary";
  btnContainer.appendChild(coffeeBtn);

  textContainer.appendChild(btnContainer);
  heroEl.appendChild(textContainer);

  // --- Marquee Section ---

  // Create a wrapper for the marquees with gradient overlays on the sides
  const marqueeWrapper = document.createElement("div");
  marqueeWrapper.style.position = "relative";
  marqueeWrapper.style.width = "80vw";
  marqueeWrapper.style.overflow = "hidden";
  marqueeWrapper.style.borderRadius = "8px";
  marqueeWrapper.style.background = "var(--color-background)";
  marqueeWrapper.style.marginTop = "2rem";

  // Use your updated marquee helper (Version 1 for image arrays) with larger images
  // Modify the helper so that it uses larger dimensions.
  // Here, we assume the createMarquee function sets image width to 300px.
  // (Alternatively, you can update the styling below.)


  // app/static/js/ui/marquee.js
  // Creates a marquee container that scrolls through an array of image objects in a continuous loop.
  // The content is duplicated so that when one set scrolls out, the next immediately appears.
  // Options:
  //   - reverse: (Boolean) scroll in the opposite direction if true.
  //   - pauseOnHover: (Boolean) pause the animation when hovering.
  function createLargerMarquee(imageArray, options = {}) {
    const { reverse = false, pauseOnHover = false } = options;

    const marqueeContainer = document.createElement('div');
    marqueeContainer.style.overflow = 'hidden';
    marqueeContainer.style.whiteSpace = 'nowrap';
    marqueeContainer.style.margin = '1rem 0';
    marqueeContainer.style.position = 'relative';

    // Create the inner container for the images.
    const inner = document.createElement('div');
    inner.style.display = 'inline-flex';
    // Force the inner container to be exactly as wide as its content.
    inner.style.width = "max-content";
    
    // Set the scrolling animation.
    inner.style.animation = reverse
      ? 'marqueeReverse 105s linear infinite'
      : 'marquee 105s linear infinite';

    // Function to append each image from the array.
    function appendImages(container) {
      imageArray.forEach(imgData => {
        const a = document.createElement('a');
        a.href = imgData.link;
        a.target = '_blank';
        const img = document.createElement('img');
        img.src = imgData.src;
        img.alt = imgData.name;
        // Increase the dimensions for larger images.
        img.style.width = '350px';
        img.style.height = '250px';
        img.style.marginRight = '1rem';
        img.style.border = '1px solid #ccc';
        img.style.borderRadius = '8px';
        img.style.transition = 'transform 0.3s';
        img.addEventListener('mouseover', () => { img.style.transform = 'scale(1.1)'; });
        img.addEventListener('mouseout', () => { img.style.transform = 'scale(1)'; });
        a.appendChild(img);
        container.appendChild(a);
      });
    }

    // Append the images twice so the scroll appears continuous.
    appendImages(inner);
    appendImages(inner);

    marqueeContainer.appendChild(inner);

    // If pauseOnHover is enabled, pause the animation when the mouse is over the marquee.
    if (pauseOnHover) {
      marqueeContainer.addEventListener('mouseover', () => {
        inner.style.animationPlayState = 'paused';
      });
      marqueeContainer.addEventListener('mouseout', () => {
        inner.style.animationPlayState = 'running';
      });
    }

    // Append the CSS keyframes for the marquee animations if not already defined.
    if (!document.getElementById('marquee-styles')) {
      const style = document.createElement('style');
      style.id = 'marquee-styles';
      style.innerHTML = `
        @keyframes marquee {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        @keyframes marqueeReverse {
          0% { transform: translateX(0); }
          100% { transform: translateX(50%); }
        }
      `;
      document.head.appendChild(style);
    }

    return marqueeContainer;
  }


  // Example usage:
  // Create two marquee rows using the updated helper:
  const marquee1 = createLargerMarquee(secondRow, { reverse: false, pauseOnHover: true });
  const marquee2 = createLargerMarquee(firstRow, { reverse: true, pauseOnHover: true });
  marqueeWrapper.appendChild(marquee1);
  marqueeWrapper.appendChild(marquee2);


  // // Create two marquee rows using our helper with larger images.
  // const marquee1 = createMarqueeForImages(firstRow, { reverse: false, pauseOnHover: true, repeat: 2 });
  // const marquee2 = createMarqueeForImages(secondRow, { reverse: true, pauseOnHover: true, repeat: 2 });
  // marqueeWrapper.appendChild(marquee1);
  // marqueeWrapper.appendChild(marquee2);

  // Create gradient overlays for fade effect at left and right edges
  const leftOverlay = document.createElement("div");
  leftOverlay.style.position = "absolute";
  leftOverlay.style.top = "0";
  leftOverlay.style.left = "0";
  leftOverlay.style.bottom = "0";
  leftOverlay.style.width = "30%";
  leftOverlay.style.background = "linear-gradient(to right, var(--color-background), transparent)";
  marqueeWrapper.appendChild(leftOverlay);

  const rightOverlay = document.createElement("div");
  rightOverlay.style.position = "absolute";
  rightOverlay.style.top = "0";
  rightOverlay.style.right = "0";
  rightOverlay.style.bottom = "0";
  rightOverlay.style.width = "30%";
  rightOverlay.style.background = "linear-gradient(to left, var(--color-background), transparent)";
  marqueeWrapper.appendChild(rightOverlay);

  heroEl.appendChild(marqueeWrapper);


  // ---------------------------------------------- Added for Mobile Screens ----------------------------------------------
  // Adjust for Mobile Screens
  function adjustHeroLayout() {
    if (window.innerWidth <= 768) {
      line1.style.fontSize = "2rem";
      line2.style.fontSize = "2.5rem";
      line3.style.fontSize = "1.2rem";
      textContainer.style.textAlign = "center";
    }
  }

  adjustHeroLayout();
  window.addEventListener("resize", adjustHeroLayout);

// ---------------------------------------------------------------------------------------------------------------
})();