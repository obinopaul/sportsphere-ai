// app/static/js/ui/marquee.js
// Creates a marquee container that scrolls through an array of image objects.
// Options:
//   - reverse: (Boolean) scroll in the opposite direction if true.
//   - pauseOnHover: (Boolean) pause the animation when hovering over the marquee.
//   - repeat: (Number) how many times the image set should repeat (default: 2).
function createMarquee(imageArray, options = {}) {
  const { reverse = false, pauseOnHover = false, repeat = 2 } = options;

  const marqueeContainer = document.createElement('div');
  marqueeContainer.style.overflow = 'hidden';
  marqueeContainer.style.whiteSpace = 'nowrap';
  marqueeContainer.style.margin = '1rem 0';
  marqueeContainer.style.position = 'relative';

  // Create inner container that will hold repeated copies of the images.
  const innerContainer = document.createElement('div');
  innerContainer.style.display = 'inline-flex'; // use inline-flex for horizontal layout
  innerContainer.style.animation = reverse
    ? 'marqueeReverse 15s linear infinite'
    : 'marquee 15s linear infinite';

  // Repeat the image content 'repeat' times
  for (let i = 0; i < repeat; i++) {
    imageArray.forEach(imgData => {
      const a = document.createElement('a');
      a.href = imgData.link;
      a.target = '_blank';
      const img = document.createElement('img');
      img.src = imgData.src;
      img.alt = imgData.name;
      img.style.width = '250px';
      img.style.marginRight = '1rem';
      img.style.border = '1px solid #ccc';
      img.style.borderRadius = '8px';
      img.style.transition = 'transform 0.3s';
      img.addEventListener('mouseover', () => { img.style.transform = 'scale(1.1)'; });
      img.addEventListener('mouseout', () => { img.style.transform = 'scale(1)'; });
      a.appendChild(img);
      innerContainer.appendChild(a);
    });
  }
  
  marqueeContainer.appendChild(innerContainer);

  // If pauseOnHover is enabled, pause the animation on mouseover.
  if (pauseOnHover) {
    marqueeContainer.addEventListener('mouseover', () => {
      innerContainer.style.animationPlayState = 'paused';
    });
    marqueeContainer.addEventListener('mouseout', () => {
      innerContainer.style.animationPlayState = 'running';
    });
  }
  
  // Append CSS keyframes for the marquee animations if not already defined.
  if (!document.getElementById('marquee-styles')) {
    const style = document.createElement('style');
    style.id = 'marquee-styles';
    style.innerHTML = `
      @keyframes marquee {
        0% { transform: translateX(0); }
        100% { transform: translateX(-100%); }
      }
      @keyframes marqueeReverse {
        0% { transform: translateX(0); }
        100% { transform: translateX(100%); }
      }
    `;
    document.head.appendChild(style);
  }
  return marqueeContainer;
}
