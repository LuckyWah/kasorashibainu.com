const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const path = require('path');
const app = express();

app.use(express.json());
app.use(cors()); // Allow requests from your GitHub Pages domain

// Connect to MongoDB
mongoose.connect('mongodb://localhost/skeb-sniper', { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => console.error('MongoDB connection error:', err));

// Coupon Schema
const couponSchema = new mongoose.Schema({
    code: { type: String, required: true, unique: true },
    discount: { type: Number, required: true },
    used: { type: Boolean, default: false }
});
const Coupon = mongoose.model('Coupon', couponSchema);

// Initialize coupons (run this once to populate the database)
async function initializeCoupons() {
    const coupons = [
        { code: "HALFOFF", discount: 25, used: false },
        { code: "trey_158", discount: 50, used: false }
    ];
    for (const coupon of coupons) {
        await Coupon.findOneAndUpdate({ code: coupon.code }, coupon, { upsert: true });
    }
}
initializeCoupons();

// Generate temporary download links (valid for 1 hour)
function generateDownloadLinks() {
    const token = Math.random().toString(36).substring(2); // Simple token for now
    const expires = Date.now() + 60 * 60 * 1000; // 1 hour expiry
    return {
        windowsUrl: `https://your-gcp-server-ip:3000/download/windows?token=${token}&expires=${expires}`,
        linuxUrl: `https://your-gcp-server-ip:3000/download/linux?token=${token}&expires=${expires}`
    };
}

// Validate coupon endpoint
app.post('/api/validate-coupon', async (req, res) => {
    const { couponCode } = req.body;
    try {
        const coupon = await Coupon.findOne({ code: couponCode });
        if (!coupon) {
            return res.json({ valid: false, message: "Invalid coupon code" });
        }
        if (coupon.used) {
            return res.json({ valid: false, message: "This coupon has already been used" });
        }
        res.json({ valid: true, discount: coupon.discount });
    } catch (error) {
        res.status(500).json({ valid: false, message: "Server error" });
    }
});

// Free access endpoint (for $0 price)
app.post('/api/free-access', async (req, res) => {
    const { couponCode } = req.body;
    try {
        const coupon = await Coupon.findOne({ code: couponCode });
        if (!coupon || coupon.used) {
            return res.json({ success: false, message: "Invalid or used coupon" });
        }
        coupon.used = true;
        await coupon.save();
        const downloadLinks = generateDownloadLinks();
        res.json({ success: true, downloadLinks });
    } catch (error) {
        res.status(500).json({ success: false, message: "Server error" });
    }
});

// Validate payment endpoint (for non-$0 payments)
app.post('/api/validate-payment', async (req, res) => {
    const { orderId, couponCode } = req.body;
    try {
        // Here you could validate the PayPal order with PayPal's API if needed
        // For now, assume the payment is valid
        if (couponCode) {
            const coupon = await Coupon.findOne({ code: couponCode });
            if (coupon && !coupon.used) {
                coupon.used = true;
                await coupon.save();
            }
        }
        const downloadLinks = generateDownloadLinks();
        res.json({ success: true, downloadLinks });
    } catch (error) {
        res.status(500).json({ success: false, message: "Server error" });
    }
});

// Serve download files with token validation
app.get('/download/:platform', (req, res) => {
    const { token, expires } = req.query;
    const { platform } = req.params;

    // Validate token and expiry (simplified)
    if (!token || !expires || Date.now() > parseInt(expires)) {
        return res.status(403).send("Link expired or invalid");
    }

    const filePath = platform === 'windows' 
        path.join(__dirname, 'downloads', 'skeb-sniper-installer.exe');
    res.download(filePath);
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));