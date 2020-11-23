"use strict";

const bars = document.querySelector(".fa.fa-bars");
const navMenu = document.querySelector(".nav-menu");
AOS.init();
/* 
const prevBtn = document.querySelector(".prev");
const nextBtn = document.querySelector(".next"); */
//Active bars
bars.addEventListener("click", function () {
  navMenu.classList.toggle("slider");
  bars.classList.toggle("timesX");
});
