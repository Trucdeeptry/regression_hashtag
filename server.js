const express = require('express');
const { trainAndPredictFromInput } = require('./lib/regresstion.js');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

app.post('/predict', async (req, res) => {
  try {
    const {new_ad} = req.body;

    const result = await trainAndPredictFromInput(new_ad);
    res.json(result);
  } catch (err) {
    console.error('❌ Lỗi dự đoán:', err);
    res.status(500).json({ error: 'Lỗi xử lý' });
  }
});

app.get('/', (req, res) => {
  res.send('👍 API Predict hoạt động');
});

app.listen(PORT, () => {
  console.log(`🚀 Server đang chạy ở cổng ${PORT}`);
});
