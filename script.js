const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(".reveal").forEach((el, idx) => {
  el.style.transitionDelay = `${Math.min(idx * 70, 300)}ms`;
  observer.observe(el);
});

const carousel = document.querySelector("[data-carousel]");

if (carousel) {
  const slides = Array.from(carousel.querySelectorAll("[data-carousel-slide]"));
  const prevBtn = carousel.querySelector("[data-carousel-prev]");
  const nextBtn = carousel.querySelector("[data-carousel-next]");
  const dotsWrap = document.querySelector("[data-carousel-dots]");

  if (!slides.length || !dotsWrap) {
    console.warn("Carousel markup incomplete: skipping carousel setup.");
  } else {

  let currentIndex = Math.max(
    0,
    slides.findIndex((slide) => slide.classList.contains("is-active"))
  );

  if (currentIndex < 0) currentIndex = 0;

  const dots = slides.map((_, idx) => {
    const dot = document.createElement("button");
    dot.className = "carousel-dot";
    dot.type = "button";
    dot.setAttribute("aria-label", `Go to pair ${idx + 1}`);
    dot.addEventListener("click", () => {
      goTo(idx);
      restartAutoPlay();
    });
    dotsWrap.appendChild(dot);
    return dot;
  });

  function render() {
    slides.forEach((slide, idx) => {
      const active = idx === currentIndex;
      slide.classList.toggle("is-active", active);
      slide.setAttribute("aria-hidden", String(!active));
    });

    dots.forEach((dot, idx) => {
      dot.classList.toggle("is-active", idx === currentIndex);
    });
  }

  function goTo(index) {
    const count = slides.length;
    currentIndex = (index + count) % count;
    render();
  }

  prevBtn?.addEventListener("click", () => {
    goTo(currentIndex - 1);
    restartAutoPlay();
  });

  nextBtn?.addEventListener("click", () => {
    goTo(currentIndex + 1);
    restartAutoPlay();
  });

  let timer = null;

  function startAutoPlay() {
    stopAutoPlay();
    timer = window.setInterval(() => {
      goTo(currentIndex + 1);
    }, 4500);
  }

  function stopAutoPlay() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
  }

  function restartAutoPlay() {
    startAutoPlay();
  }

  carousel.addEventListener("mouseenter", stopAutoPlay);
  carousel.addEventListener("mouseleave", startAutoPlay);

  render();
  startAutoPlay();
  }
}
