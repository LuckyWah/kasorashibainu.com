// ------------------------------------ Show navigation bar ------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    // Show/hide navigation bar on click
    function toggleMenu() {
        const navTabs = document.querySelector(".nav-tabs");
        navTabs.classList.toggle("show");
    }

    // Add event listener to the hamburger menu button
    document.querySelector(".menu-toggle").addEventListener("click", toggleMenu);

    // Automatically close menu when clicking a nav link (on mobile)
    document.querySelectorAll(".nav-tabs button").forEach(button => {
        button.addEventListener("click", () => {
            const navTabs = document.querySelector(".nav-tabs");
            if (navTabs.classList.contains("show")) {
                navTabs.classList.remove("show"); // Hide menu after clicking
            }
        });
    });
});

// ------------------------------------ Tab switching ------------------------------------
function openTab(evt, tabName) {
    const tabcontent = document.getElementsByClassName("tabcontent");
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].classList.remove("active");
        tabcontent[i].innerHTML = "";
    }

    const tablinks = document.getElementsByClassName("tablinks");
    for (let j = 0; j < tablinks.length; j++) {
        tablinks[j].classList.remove("active");
    }

    let targetTab = document.getElementById(tabName);
    if (!targetTab) {
        targetTab = document.createElement("div");
        targetTab.id = tabName;
        targetTab.classList.add("tabcontent");
        const container = document.querySelector(".container");
        if (container) container.appendChild(targetTab);
        else {
            console.error("Container element not found!");
            return;
        }
    }

    // Add a loading message while fetching
    targetTab.innerHTML = `<p>Loading ${tabName}...</p>`;
    targetTab.classList.add("active");

    fetch(tabName + ".html", { cache: "no-cache" })
        .then(response => {
            if (!response.ok) throw new Error(`Failed to load content (${response.status})`);
            return response.text();
        })
        .then(data => {
            if (!data.trim()) throw new Error(`Empty content for ${tabName}.html`);
            const doc = new DOMParser().parseFromString(data, "text/html");
            const fetchedContent = doc.getElementById(tabName);
            if (fetchedContent) {
                targetTab.innerHTML = fetchedContent.innerHTML;
                targetTab.classList.add("active");
                if (tabName === "home") {
                    initializeCarousel();
                    initFeatureCardLayout();
                }
                if (tabName === "contact") attachFormHandler();
                if (tabName === "purchase") initializePurchaseTab();
            } else {
                throw new Error(`Tab content not found in ${tabName}.html`);
            }
        })
        .catch(error => {
            console.error(`Error loading tab '${tabName}':`, error);
            targetTab.innerHTML = `<p>Failed to load ${tabName} content. Please try again later.</p>`;
            targetTab.classList.add("active");
        });

    if (evt?.currentTarget) {
        evt.currentTarget.classList.add("active");
    } else if (tabName === "home") {
        const homeBtn = document.querySelector(".tablinks[onclick*='openTab(event, \"home\")']");
        if (homeBtn) homeBtn.classList.add("active");
    }

    history.replaceState(null, "", `#${tabName}`);
}

// ------------------------------------ Load tab with redirect check ------------------------------------
window.onload = function () {
    // If the page is loaded with a direct URL (e.g., /contact), ensure index.html is loaded first
    if (!document.querySelector(".container")) {
        window.location.replace("/skeb_sniper/index.html" + window.location.hash);
        return;
    }

    let path = window.location.hash.substring(1) || "home"; // Default to "home"
    const validTabs = ["home", "purchase", "contact", "faq", "trust", "tos", "privacy", "release", "about"];
    openTab(null, validTabs.includes(path) ? path : "home");
};

// Handle back/forward navigation
window.onpopstate = function () {
    let path = window.location.hash.substring(1) || "home"; // Default to "home"
    openTab(null, path);
};

// Carousel Initialization
function initializeCarousel() {
    let currentSlide = 0;
    const slides = document.getElementsByClassName("slide");
    const dots = document.getElementsByClassName("dot");

    function showSlide(index) {
        currentSlide = index >= slides.length ? 0 : index < 0 ? slides.length - 1 : index;
        for (let i = 0; i < slides.length; i++) {
            slides[i].classList.remove("active");
            if (dots[i]) dots[i].classList.remove("active");
        }
        slides[currentSlide].classList.add("active");
        if (dots[currentSlide]) dots[currentSlide].classList.add("active");
    }

    window.changeSlide = n => showSlide(currentSlide + n);
    window.jumpToSlide = n => showSlide(n);

    const featureHeading = document.getElementById("featureHeading");
    const featureCarousel = document.getElementById("featureCarousel");
    const adAndCarousel = document.getElementById("adAndCarousel");

    if (featureHeading && featureCarousel && adAndCarousel) {
        // Function to set height
        function setContainerHeight() {
            adAndCarousel.getBoundingClientRect(); // Force reflow
            const newHeight = adAndCarousel.scrollHeight + "px";
            adAndCarousel.style.maxHeight = newHeight;
        }

        // Wait for advertisement animation to complete
        setTimeout(() => {
            setContainerHeight();
            // Animate carousel into view
            featureCarousel.style.opacity = "1";
            featureCarousel.style.transform = "translateX(0)";
            featureCarousel.style.pointerEvents = "auto";

            // Recheck height after another delay to catch late renders
            setTimeout(setContainerHeight, 500);}, 1500); // Increased delay to 1.5s for safety

        showSlide(0);
    } else {
        console.warn("Carousel elements missing.");
    }
}

// ------------------------------------ Feature Card Width Adjustment ------------------------------------
function initFeatureCardLayout() {
    const containers = document.querySelectorAll('.features-container, .feature-box-container');
    containers.forEach(container => {
      const updateWidth = () => {
        const containerWidth = container.offsetWidth;
        const gap = 20;
        const min = 300;
        const max = 350;
  
        const maxCount = Math.floor((containerWidth + gap) / (min + gap));
        const minCount = Math.ceil((containerWidth + gap) / (max + gap));
        const count = Math.max(1, Math.min(maxCount, minCount));
        const itemWidth = (containerWidth - (count - 1) * gap) / count;
  
        container.style.setProperty('--card-width', `${itemWidth}px`);
      };
  
      updateWidth();
      window.addEventListener('resize', updateWidth);
    });
  }
  
// ------------------------------------ Chatbot ------------------------------------
function toggleChat() {
    const box = document.getElementById('chatbox');
    box.style.display = box.style.display === 'none' || !box.style.display ? 'block' : 'none';
}

function formatMessage(text) {
    // Handle inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold (**bold**)
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic (_italic_)
    text = text.replace(/_(.+?)_/g, '<em>$1</em>');
    return text;
  }

async function sendMessage() {
    const input = document.getElementById('input');
    const chat = document.getElementById('chat');
    const message = input.value.trim();
    if (!message) return;

    const userMsg = document.createElement('div');
    userMsg.className = 'message user';
    userMsg.textContent = message;
    chat.appendChild(userMsg);
    chat.scrollTop = chat.scrollHeight;
    input.value = '';

    const loadingId = 'loading-' + Date.now();
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'message loading';
    loadingMsg.id = loadingId;
    loadingMsg.textContent = 'Trying to respond...';
    chat.appendChild(loadingMsg);
    chat.scrollTop = chat.scrollHeight;

    try {
        const res = await fetch('https://download.kasorashibainu.com/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        const data = await res.json();
        document.getElementById(loadingId).remove();
        
        const botMsg = document.createElement('div');
        botMsg.className = 'message bot';
        botMsg.innerHTML = formatMessage(data.response || "You've reached the limit. Please come back later.");
        chat.appendChild(botMsg);
        chat.scrollTop = chat.scrollHeight;
    } catch (err) {
        document.getElementById(loadingId).remove();
        const errorMsg = document.createElement('div');
        errorMsg.className = 'message error';
        errorMsg.textContent = 'Error contacting chatbot';
        chat.appendChild(errorMsg);
        chat.scrollTop = chat.scrollHeight;
    }
}

function handleKey(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        sendMessage();
    }
}
  
// ------------------------------------ Form handler ------------------------------------
function attachFormHandler() {
    const Form = document.getElementById("Form");
    const ResponseEl = document.getElementById("Response");

    if (Form && ResponseEl) {
        Form.addEventListener("submit", function(e) {
            e.preventDefault();
            console.log("Form submission intercepted");

            const formData = new FormData(Form);
            const submitButton = Form.querySelector("button[type='submit']");
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerText = "Submitting...";
            }

            fetch(Form.action, {
                method: "POST",
                headers: { "Accept": "application/json" },
                body: formData
            })
            .then(response => {
                console.log("Fetch response:", response);
                if (response.ok) {
                    return response.json().then(() => {
                        ResponseEl.innerText = "Thank you! Your message has been submitted.";
                        ResponseEl.style.color = "green";
                        setTimeout(() => {
                            Form.reset(); // Reset the form but do not reload the page
                            submitButton.disabled = false;
                            submitButton.innerText = "Submit";
                        }, 3000);
                    });
                } else {
                    return response.json().then(data => {
                        ResponseEl.innerText = data.errors
                            ? data.errors.map(error => error.message).join(", ")
                            : "Oops! There was a problem submitting your form.";
                        ResponseEl.style.color = "red";
                    });
                }
            })
            .catch(error => {
                console.error("Fetch error:", error);
                ResponseEl.innerText = "Oops! There was a problem submitting your form.";
                ResponseEl.style.color = "red";
            })
            .finally(() => {
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.innerText = "Submit";
                }
            });
        });
    } else {
        console.error("Form or Response element not found in contact tab");
    }
}

// ------------------------------------ Purchase function ------------------------------------
// Global state for the purchase tab
const purchaseState = {
    selectedPlan: null, // 'monthly' or 'yearly'
    couponCode: null,
    isCouponValid: false, // Track if a valid coupon is applied
    effectivePlan: null, // Track the actual plan used for subscription (after coupon)
    downloadLinks: null,
    licenseKey: null
};

// Update the discount message element
function updateDiscountMessage(discountMessage, message, color) {
    discountMessage.innerText = message;
    discountMessage.style.color = color;
}

// Function to check if a coupon has been used
function isCouponUsed(couponCode) {
    const usedCoupons = JSON.parse(localStorage.getItem("usedCoupons") || "[]");
    return usedCoupons.includes(couponCode);
}

// Function to mark a coupon as used
function markCouponAsUsed(couponCode) {
    let usedCoupons = JSON.parse(localStorage.getItem("usedCoupons") || "[]");
    if (!usedCoupons.includes(couponCode)) {
        usedCoupons.push(couponCode);
        localStorage.setItem("usedCoupons", JSON.stringify(usedCoupons));
    }
}

// Render the download section
function renderDownloadSection(downloadContainer, downloadLinks, licenseKey) {
    if (!downloadContainer) return;

    const downloadBtnWindows = downloadContainer.querySelector("#download-btn-windows");
    const downloadBtnLinux = downloadContainer.querySelector("#download-btn-linux");
    const instructions = downloadContainer.querySelector("p");

    // Set up Windows download link
    if (downloadBtnWindows) {
        downloadBtnWindows.href = downloadLinks.windows; // e.g., "/api/download-windows?userId=123"
        downloadBtnWindows.download = "skeb-sniper-windows.exe"; // Suggest filename
        downloadBtnWindows.style.display = "inline-block";
    }

    // Set up Linux download link
    if (downloadBtnLinux) {
        downloadBtnLinux.href = downloadLinks.linux; // e.g., "/api/download-linux?userId=123"
        downloadBtnLinux.download = "skeb-sniper-linux.tar";
        downloadBtnLinux.style.display = "inline-block";
    }

    // License section (unchanged)
    const licenseSection = document.createElement("div");
    licenseSection.id = "license-section";
    licenseSection.className = "license-section";

    const licenseLabel = document.createElement("span");
    licenseLabel.innerText = `Your License Key: ${licenseKey}`;

    const copyButton = document.createElement("button");
    copyButton.innerText = "Copy";
    copyButton.onclick = () => {
        navigator.clipboard.writeText(licenseKey).then(() => {
            copyButton.innerText = "Copied!";
            setTimeout(() => copyButton.innerText = "Copy", 2000);
        }).catch(err => {
            console.error("Failed to copy license key:", err);
            alert("Failed to copy license key. Please copy it manually.");
        });
    };

    licenseSection.appendChild(licenseLabel);
    licenseSection.appendChild(copyButton);
    downloadContainer.appendChild(licenseSection);

    // Instructions and UI updates
    instructions.innerHTML = 'Download the Windows or Linux version below. For Linux users, refer to the FAQ section for detailed instructions. Ensure Docker is installed. Use the license key above to activate the software.';
    downloadContainer.style.display = "block";
    document.getElementById("purchase-button").style.display = "none";
    document.getElementById("coupon-section").style.display = "none";
    document.getElementById("license-check-section").style.display = "none";
    document.getElementById("already-purchased-btn").style.display = "none";
    document.querySelector("h2").style.display = "none";
    document.querySelectorAll("p").forEach(p => {
        if (p.querySelector("strong") && p.textContent.includes("Note for Linux Users")) return;
        p.style.display = "none";
    });
    document.querySelector(".plan-selection").style.display = "none";
}

// Render PayPal subscription button based on selected plan and coupon
function renderPayPalButtons(purchaseButtonContainer) {
    const { selectedPlan, isCouponValid } = purchaseState;

    if (!selectedPlan) {
        console.error("No plan selected.");
        updateDiscountMessage(document.getElementById("discount-message"), "Please select a plan to proceed.", "red");
        return;
    }

    let effectivePlan = selectedPlan;
    let planId;
    let planDescription;

    if (selectedPlan === 'monthly') {
        if (isCouponValid) {
            effectivePlan = 'monthly-free-3-months';
            planId = 'P-22X515193L502430DM7RQSKY';
            planDescription = 'First 3 months free, then $1/month';
        } else {
            planId = 'P-38M666889H159984WM76HWKI';
            planDescription = '30-day free trial, then $1/month';
        }
    } else {
        planId = 'P-9MY49149LD6839715M7QJAYA';
        planDescription = '$10/year';
    }

    purchaseState.effectivePlan = effectivePlan;

    purchaseButtonContainer.innerHTML = `
        <div class="plan-selection">
            <h3>Your Selected Plan: ${effectivePlan === 'monthly-free-3-months' ? 'Monthly Plan (First 3 Months Free)' : (selectedPlan.charAt(0).toUpperCase() + selectedPlan.slice(1))}</h3>
            <div class="plan-option">
                <strong>${effectivePlan === 'monthly-free-3-months' ? 'Monthly Plan (First 3 Months Free)' : (selectedPlan === 'monthly' ? 'Monthly Plan' : 'Yearly Plan')}</strong>
                <p>${planDescription}</p>
                <div id="paypal-button-container-${planId}"></div>
            </div>
        </div>
    `;
    purchaseButtonContainer.style.display = "block";

    if (typeof paypal === 'undefined' || !paypal.Buttons) {
        console.error("PayPal SDK not loaded.");
        purchaseButtonContainer.innerHTML = `
            <div class="error-message" style="color: red; font-weight: bold;">
                Error: Unable to load PayPal payment system. Please refresh the page or contact support.
            </div>
        `;
        updateDiscountMessage(document.getElementById("discount-message"), "PayPal SDK failed to load.", "red");
        return;
    }

    paypal.Buttons({
        style: {
            shape: 'rect',
            color: 'gold',
            layout: 'vertical',
            label: 'subscribe'
        },
        createSubscription: function(data, actions) {
            return actions.subscription.create({
                plan_id: planId
            });
        },
        onApprove: function(data, actions) {
            handleSubscriptionSuccess(data.subscriptionID, effectivePlan);
        },
        onError: function(err) {
            console.error(`PayPal error (${effectivePlan}):`, err);
            updateDiscountMessage(document.getElementById("discount-message"), `Error processing ${effectivePlan} subscription.`, "red");
        }
    }).render(`#paypal-button-container-${planId}`);
}

// Handle subscription success
async function handleSubscriptionSuccess(subscriptionID, effectivePlan) {
    try {
        const response = await fetch('https://download.kasorashibainu.com/api/validate-subscription', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                subscriptionId: subscriptionID, 
                plan: effectivePlan,
                couponCode: purchaseState.couponCode || ""
            })
        });
        const data = await response.json();
        if (data.success) {
            purchaseState.downloadLinks = data.downloadLinks;
            purchaseState.licenseKey = data.license_key;
            localStorage.setItem("subscriptionCompleted", "true");
            localStorage.setItem("downloadLinks", JSON.stringify(data.downloadLinks));
            localStorage.setItem("licenseKey", data.license_key);
            localStorage.setItem("subscriptionId", subscriptionID);
            renderDownloadSection(document.getElementById("download-container"), purchaseState.downloadLinks, purchaseState.licenseKey);
            updateDiscountMessage(document.getElementById("discount-message"), `Subscribed successfully to ${effectivePlan === 'monthly-free-3-months' ? 'Monthly Plan (First 3 Months Free)' : effectivePlan + ' plan'}! Download below.`, "green");
        } else {
            updateDiscountMessage(document.getElementById("discount-message"), "Subscription validation failed: " + data.message, "red");
        }
    } catch (error) {
        console.error("Error validating subscription:", error);
        updateDiscountMessage(document.getElementById("discount-message"), "Error validating subscription.", "red");
    }
}

// Handle coupon application (only for monthly plan)
async function handleCouponClick() {
    const couponInput = document.getElementById("coupon");
    const discountMessage = document.getElementById("discount-message");
    const couponCode = couponInput.value.trim();
    purchaseState.couponCode = couponCode;

    if (couponCode) {
        updateDiscountMessage(discountMessage, "Validating coupon...", "gray");
        try {
            const response = await fetch('https://download.kasorashibainu.com/api/validate-coupon', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ couponCode })
            });
            const data = await response.json();
            if (data.valid) {
                purchaseState.isCouponValid = true;
                updateDiscountMessage(discountMessage, "Coupon applied: First 3 months free, then $1/month!", "green");
                markCouponAsUsed(couponCode);
            } else {
                purchaseState.isCouponValid = false;
                purchaseState.couponCode = null;
                updateDiscountMessage(discountMessage, data.message || "Invalid coupon code.", "red");
            }
        } catch (error) {
            console.error("Error validating coupon:", error);
            purchaseState.isCouponValid = false;
            purchaseState.couponCode = null;
            updateDiscountMessage(discountMessage, "Error validating coupon.", "red");
        }
    } else {
        purchaseState.isCouponValid = false;
        purchaseState.couponCode = null;
        updateDiscountMessage(discountMessage, "No coupon applied.", "black");
    }

    const purchaseButtonContainer = document.getElementById("purchase-button");
    renderPayPalButtons(purchaseButtonContainer);
}

// Handle license check
async function handleLicenseCheck(subscriptionId, effectivePlan) {
    try {
        const response = await fetch('https://download.kasorashibainu.com/api/validate-subscription', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                subscriptionId: subscriptionId, 
                plan: effectivePlan || 'monthly',
                couponCode: ""
            })
        });

        const licenseErrorMessage = document.getElementById("license-error-message");

        if (!response.ok) {
            const errorData = await response.json();
            const errorMessage = errorData.message || "Failed to retrieve license. Please double-check your Subscription ID.";
            licenseErrorMessage.textContent = errorMessage;
            licenseErrorMessage.style.color = "red";
            licenseErrorMessage.style.display = "block";
            return;
        }

        const data = await response.json();

        if (data.success) {
            licenseErrorMessage.style.display = "none";  // Hide error if success

            purchaseState.downloadLinks = data.downloadLinks;
            purchaseState.licenseKey = data.license_key;
            localStorage.setItem("subscriptionCompleted", "true");
            localStorage.setItem("downloadLinks", JSON.stringify(data.downloadLinks));
            localStorage.setItem("licenseKey", data.license_key);
            localStorage.setItem("subscriptionId", subscriptionId);
            renderDownloadSection(document.getElementById("download-container"), purchaseState.downloadLinks, purchaseState.licenseKey);
        } else {
            const errorMessage = data.message || "Failed to retrieve license. Please double-check your Subscription ID.";
            licenseErrorMessage.textContent = errorMessage;
            licenseErrorMessage.style.color = "red";
            licenseErrorMessage.style.display = "block";
        }
    } catch (error) {
        console.error("Error retrieving license:", error);
        const licenseErrorMessage = document.getElementById("license-error-message");
        licenseErrorMessage.textContent = "Network error. Please try again or contact support.";
        licenseErrorMessage.style.color = "red";
        licenseErrorMessage.style.display = "block";
    }
}

// Initialize the Purchase tab
function initializePurchaseTab() {
    const purchaseTab = document.getElementById("purchase");
    if (!purchaseTab) return;

    const couponSection = document.getElementById("coupon-section");
    const planOptions = document.querySelectorAll(".plan-option");
    const licenseCheckSection = document.getElementById("license-check-section");
    const alreadyPurchasedBtn = document.getElementById("already-purchased-btn");
    const purchaseContent = document.querySelector(".purchase-content");
    const showPurchaseSectionLink = document.getElementById("show-purchase-section");

    // Check if a subscription was already completed
    const subscriptionCompleted = localStorage.getItem("subscriptionCompleted");
    if (subscriptionCompleted === "true") {
        const downloadLinks = JSON.parse(localStorage.getItem("downloadLinks"));
        const licenseKey = localStorage.getItem("licenseKey");
        if (downloadLinks && licenseKey) {
            renderDownloadSection(document.getElementById("download-container"), downloadLinks, licenseKey);
            updateDiscountMessage(
                document.getElementById("discount-message"),
                "Subscription already completed. Download below.",
                "green"
            );
            return; // Skip the rest of the initialization
        }
    }

    // Default to showing the purchase section
    document.getElementById("purchase-button").style.display = "none";
    document.getElementById("coupon-section").style.display = "none";
    document.getElementById("download-container").style.display = "none";
    licenseCheckSection.style.display = "none";
    purchaseContent.style.display = "block";

    // Add event listener for "Already Purchased?" button
    if (alreadyPurchasedBtn) {
        alreadyPurchasedBtn.addEventListener("click", () => {
            licenseCheckSection.style.display = "block";
            purchaseContent.style.display = "none";
            document.getElementById("purchase-button").style.display = "none";
            document.getElementById("coupon-section").style.display = "none";
            updateDiscountMessage(document.getElementById("discount-message"), "", "black");
        });
    }

    // Add event listener for license check
    const checkLicenseBtn = document.getElementById("check-license-btn");
    if (checkLicenseBtn) {
        checkLicenseBtn.addEventListener("click", () => {
            const subscriptionId = document.getElementById("subscription-id").value.trim();
            if (!subscriptionId) {
                updateDiscountMessage(document.getElementById("discount-message"), "Please enter both email and subscription ID.", "red");
                return;
            }
            handleLicenseCheck(subscriptionId, purchaseState.effectivePlan);
        });
    }

    // Add event listener to show purchase section
    if (showPurchaseSectionLink) {
        showPurchaseSectionLink.addEventListener("click", (e) => {
            e.preventDefault();
            licenseCheckSection.style.display = "none";
            purchaseContent.style.display = "block";
            document.getElementById("purchase-button").style.display = purchaseState.selectedPlan ? "block" : "none";
            document.getElementById("coupon-section").style.display = purchaseState.selectedPlan === 'monthly' ? "block" : "none";
            updateDiscountMessage(document.getElementById("discount-message"), "", "black");
        });
    }

    function renderSections() {
        const purchaseButtonContainer = document.getElementById("purchase-button");

        if (purchaseState.selectedPlan === 'monthly') {
            couponSection.style.display = "block";
            renderPayPalButtons(purchaseButtonContainer);
        } else if (purchaseState.selectedPlan === 'yearly') {
            couponSection.style.display = "none";
            renderPayPalButtons(purchaseButtonContainer);
        } else {
            couponSection.style.display = "none";
            purchaseButtonContainer.style.display = "none";
            document.getElementById("download-container").style.display = "none";
        }
    }

    planOptions.forEach(option => {
        option.addEventListener("click", () => {
            const plan = option.dataset.plan;
            purchaseState.selectedPlan = plan;
            
            document.getElementById(`${plan}-plan`).checked = true;
            
            planOptions.forEach(opt => opt.classList.remove("selected"));
            option.classList.add("selected");
            
            purchaseState.isCouponValid = false;
            purchaseState.couponCode = null;
            const discountMessage = document.getElementById("discount-message");
            if (discountMessage) {
                updateDiscountMessage(discountMessage, "", "black");
            }
            const couponInput = document.getElementById("coupon");
            if (couponInput) {
                couponInput.value = "";
            }

            document.getElementById("download-container").style.display = "none";
            renderSections();
        });
    });

    const applyCouponButton = document.getElementById("apply-coupon");
    if (applyCouponButton) {
        applyCouponButton.addEventListener("click", handleCouponClick);
    }
}

document.addEventListener("DOMContentLoaded", initializePurchaseTab);
