// 






























let currentSlide = 0;
const slides = document.querySelectorAll('.slide');

// Function to show the correct slide in the slider
function showSlide(index) {
    slides.forEach(slide => slide.classList.remove('active'));
    slides[index].classList.add('active');
}

// Function to go to the next slide
function nextSlide() {
    currentSlide = (currentSlide + 1) % slides.length;
    showSlide(currentSlide);
}

// Show the first slide initially and start automatic sliding
showSlide(currentSlide);
setInterval(nextSlide, 5000); // Change slide every 5 seconds
// Get the modal and image
var modal = document.getElementById("imageModal");
var modalImg = document.getElementById("modalImage");
var captionText = document.getElementById("caption");

// Get all the images in the precautions section
var images = document.querySelectorAll('.precaution-image');

// Add click event listener to each image
images.forEach(function(image) {
    image.onclick = function() {
        modal.style.display = "block"; // Show the modal
        modalImg.src = this.src; // Set the image source in the modal
        captionText.innerHTML = this.alt; // Set the caption text (optional)
    };
});

// Get the close button
var closeBtn = document.getElementsByClassName("close")[0];

// Add click event listener to close the modal
closeBtn.onclick = function() {
    modal.style.display = "none"; // Hide the modal
};

// Close the modal when clicking anywhere outside of the modal
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
};
// Adding interaction for collapsible content
const tipItems = document.querySelectorAll('.tip-item');
tipItems.forEach(item => {
    item.addEventListener('click', () => {
        item.classList.toggle('active');
    });
});



document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("prediction-form");

    form.addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent page reload

        // Collect form data
        let formData = {
            gender: document.getElementById("gender").value,
            age: document.getElementById("age").value,
            hypertension: document.getElementById("hypertension").value,
            heart_disease: document.getElementById("heart_disease").value,
            smoking_history: document.getElementById("smoking_history").value,
            bmi: document.getElementById("bmi").value,
            hba1c_level: document.getElementById("hba1c_level").value,
            blood_glucose_level: document.getElementById("blood_glucose_level").value
        };

        // Validate required fields
        if (!formData.age || !formData.bmi || !formData.blood_glucose_level) {
            alert("Please fill in all required fields.");
            return;
        }

        // Send data to Flask backend
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            let resultDiv = document.getElementById("prediction-result");
            resultDiv.style.display = "block";

            if (data.error) {
                document.getElementById("result-message").innerHTML = `<strong>Error:</strong> ${data.error}`;
            } else {
                document.getElementById("result-message").innerHTML = `
                    <strong>Prediction:</strong> ${data.prediction} <br>
                    <strong>Probability:</strong> ${data.probability}
                `;
            }
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });

    // Function to handle section visibility
    function navigateToSection(targetId) {
        document.querySelectorAll('.section').forEach(section => {
            section.style.display = 'none';
        });

        if (targetId === 'home') {
            document.querySelector('.slider').style.display = 'block';
            document.querySelector('.features').style.display = 'block';
        } else {
            document.querySelector('.slider').style.display = 'none';
            document.querySelector('.features').style.display = 'none';

            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.style.display = 'block';
                targetSection.scrollIntoView({ behavior: 'smooth' });
            }
        }
    }

    // Add event listeners to navbar links
    document.querySelectorAll('.navbar a').forEach(link => {
        link.addEventListener('click', function(event) {
            event.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            navigateToSection(targetId);
        });
    });

    // Add event listeners to "Learn More" buttons
    document.querySelectorAll('.learn-more').forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            navigateToSection(targetId);
        });
    });

    // Show home section initially
    navigateToSection('home');
   
    
    // ==========================
    // 🔥 Feature Click → Highlight Table Column
    // ==========================
    const featureCards = document.querySelectorAll(".clickable-feature");
    const tableHeaders = document.querySelectorAll("th");
    const tableCells = document.querySelectorAll("td");

    featureCards.forEach(card => {
        card.addEventListener("click", function () {
            // Remove previous highlights
            tableHeaders.forEach(th => th.classList.remove("highlight-column"));
            tableCells.forEach(td => td.classList.remove("highlight-column"));

            // Get the feature name
            const featureName = card.getAttribute("data-feature");

            // Find the corresponding column in the table
            tableHeaders.forEach((th, index) => {
                if (th.getAttribute("data-column") === featureName) {
                    // Highlight header
                    th.classList.add("highlight-column");

                    // Highlight corresponding column cells
                    document.querySelectorAll(`tbody tr`).forEach(row => {
                        row.children[index].classList.add("highlight-column");
                    });

                    // Scroll to the dataset section
                    document.querySelector(".dataset-card").scrollIntoView({ behavior: "smooth" });
                }
            });
        });
    });
});
