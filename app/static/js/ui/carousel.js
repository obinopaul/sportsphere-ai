// app/static/js/ui/carousel.js
// A simple carousel that shows an array of images with previous/next controls.
function createCarousel({ images = [], containerClass = "" }) {
  const carouselContainer = document.createElement("div");
  carouselContainer.className = "carousel " + containerClass;
  
  const track = document.createElement("div");
  track.className = "carousel-track";
  images.forEach(imgData => {
    const a = document.createElement("a");
    a.href = imgData.link;
    a.target = "_blank";
    const img = document.createElement("img");
    img.src = imgData.src;
    img.alt = imgData.name;
    img.className = "carousel-image";
    a.appendChild(img);
    track.appendChild(a);
  });
  carouselContainer.appendChild(track);
  
  // Create previous and next buttons
  const prevBtn = createButton({
    variant: "outline",
    size: "icon",
    text: "←",
    ariaLabel: "Previous slide",
    onClick: () => {
      if (currentIndex > 0) {
        currentIndex--;
        updateCarousel();
      }
    }
  });
  prevBtn.classList.add("carousel-prev");
  
  const nextBtn = createButton({
    variant: "outline",
    size: "icon",
    text: "→",
    ariaLabel: "Next slide",
    onClick: () => {
      if (currentIndex < images.length - 1) {
        currentIndex++;
        updateCarousel();
      }
    }
  });
  nextBtn.classList.add("carousel-next");
  
  carouselContainer.appendChild(prevBtn);
  carouselContainer.appendChild(nextBtn);
  
  let currentIndex = 0;
  function updateCarousel() {
    const imageWidth = track.children[0] ? track.children[0].offsetWidth : 0;
    track.style.transform = "translateX(" + (-currentIndex * imageWidth) + "px)";
  }
  
  // Update on window resize
  window.addEventListener("resize", updateCarousel);
  
  return carouselContainer;
}
