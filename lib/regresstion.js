const fs = require('fs');
const csv = require('csv-parse');
const moment = require('moment');
const MLR = require('ml-regression-multivariate-linear');
const natural = require('natural');
const seedrandom = require('seedrandom'); // ✅ Thêm dòng này

function cleanJsonString(str) {
  if (!str) return '[]';
  // Thay "" thành "
  let cleaned = str.replace(/""/g, '"');
  // Bỏ dấu phẩy cuối cùng trước dấu ] hoặc }
  cleaned = cleaned.replace(/,(\s*[\]\}])/g, '$1');
  return cleaned;
}

async function loadPostsData(filePath) {
  return new Promise((resolve, reject) => {
    const data = [];
    fs.createReadStream(filePath)
      .pipe(csv.parse({ columns: true }))
      .on('data', (row) => {
        const createdAt = moment(row.created_at, moment.ISO_8601, true);
        if (!createdAt.isValid()) return;

        let likesCount = 0;
        try {
          const likesJsonStr = cleanJsonString(row.likes);
          const likesArray = JSON.parse(likesJsonStr);
          if (Array.isArray(likesArray)) {
            likesCount = likesArray.length;
          }
        } catch (e) {
          likesCount = 0; // nếu lỗi parse, giữ 0
        }


        let shareCount = 0;
        try {
          const shareJsonStr = cleanJsonString(row.share);
          const shareArray = JSON.parse(shareJsonStr);
          if (Array.isArray(shareArray)) {
            shareCount = shareArray.length;
          }
        } catch (e) {
          shareCount = 0;
        }

        data.push({
          id: row.id,
          hashtags: row.hashtags || '',
          hour: createdAt.hour(),
          dayofweek: createdAt.day(),
          likes: likesCount,
          share: shareCount,
        });
      })
      .on('end', () => resolve(data))
      .on('error', reject);
      
  });
}


// Hàm đọc và xử lý dữ liệu từ comments_rows.csv
async function loadCommentsData(filePath) {
  return new Promise((resolve, reject) => {
    const commentCounts = {};
    fs.createReadStream(filePath)
      .pipe(csv.parse({ columns: true }))
      .on('data', (row) => {
        const postId = row.post_id;
        commentCounts[postId] = (commentCounts[postId] || 0) + 1;
      })
      .on('end', () => resolve(commentCounts))
      .on('error', reject);
  });
}

function computeTfidf(hashtags, allDocs) {
  const tfidf = new natural.TfIdf();
  allDocs.forEach(doc => tfidf.addDocument(doc));
  const features = [];
  tfidf.tfidfs(hashtags, (i, measure) => features.push(measure));
  return features.length > 0 ? features.slice(0, 5) : [0, 0, 0, 0, 0];
}

function getHashtagPairs(hashtagsString) {
  const hashtags = hashtagsString
    .split(/\s+/)
    .map(h => h.trim().toLowerCase())
    .filter(h => h.length > 0);
  const pairs = [];
  for (let i = 0; i < hashtags.length; i++) {
    for (let j = i + 1; j < hashtags.length; j++) {
      const pair = [hashtags[i], hashtags[j]].sort().join('|');
      pairs.push(pair);
    }
  }
  return pairs;
}

function computePairCommentsSum(data) {
  const pairComments = {};
  data.forEach(row => {
    const pairs = getHashtagPairs(row.hashtags);
    pairs.forEach(pair => {
      pairComments[pair] = (pairComments[pair] || 0) + row.comments;
    });
  });
  return pairComments;
}

function prepareFeatures(data, allHashtags, pairCommentsSum) {
  return data.map(row => {
    const tfidf = computeTfidf(row.hashtags, allHashtags);
    const pairs = getHashtagPairs(row.hashtags);
    let totalPairComments = 0;
    pairs.forEach(pair => {
      totalPairComments += pairCommentsSum[pair] || 0;
    });

    let hourBoost = 1;
    if ([19, 20, 21].includes(row.hour)) {
      hourBoost = 1.15;
    }

    return [
      ...tfidf,
      row.hour,
      row.dayofweek,
      totalPairComments,
      hourBoost
    ];
  });
}

// ✅ Cập nhật: truyền rng làm tham số
function shuffleArray(arr, rng = Math.random) {
  const array = arr.slice();
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// ✅ Cập nhật: truyền rng vào trainTestSplit
function trainTestSplit(data, testSize = 0.2, rng = Math.random) {
  const shuffled = shuffleArray(data, rng);
  const testCount = Math.floor(data.length * testSize);
  return {
    train: shuffled.slice(0, -testCount),
    test: shuffled.slice(-testCount)
  };
}
async function calculateGoldenHours(postsData, commentCounts) {
  const data = postsData.map(post => ({
    ...post,
    comments: commentCounts[post.id] || 0
  }));

  const interactionByHour = {};
  for (let i = 0; i < 24; i++) interactionByHour[i] = 0;

  data.forEach(row => {
    const totalInteraction = row.likes + row.share + row.comments;
    interactionByHour[row.hour] += totalInteraction;
  });

  const goldenHours = Object.entries(interactionByHour)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([hour]) => Number(hour));

  return goldenHours
}
// Hàm chính
async function trainAndPredictFromInput(new_ad) {
  try {
    const postsData = await loadPostsData('./data/posts_rows.csv');
    const commentCounts = await loadCommentsData('./data/comments_rows.csv');
    if (postsData.length === 0) throw new Error('Không có dữ liệu bài đăng hợp lệ');

    const data = postsData.map(row => ({
      ...row,
      comments: commentCounts[row.id] || 0
    }));

    const dataWithLikesAndComments = data.filter(row => row.comments > 0 && row.likes > 0);
    const dataWithoutComments = data.filter(row => row.comments === 0);
    const sampledWithoutComments = dataWithoutComments.slice(0, Math.min(50, dataWithoutComments.length));
    const balancedData = [...dataWithLikesAndComments, ...sampledWithoutComments];

    const allHashtags = balancedData.map(row => row.hashtags);

  

    // ✅ Dùng seed từ new_ad để tái lập kết quả
    // const rng = seedrandom(new_ad.created_at + new_ad.hashtags);

    let likesSum = 0, sharesSum = 0, commentsSum = 0;
    const runs = 1;

    for (let run = 0; run < runs; run++) {
      const rng = seedrandom(new_ad.created_at + new_ad.hashtags + run);
      const { train: X_train } = trainTestSplit(balancedData, 0.2, rng);

      const y_likes_train = X_train.map(d => d.likes);
      const y_share_train = X_train.map(d => d.share);
      const y_comments_train = X_train.map(d => d.comments);

      const pairCommentsSum = computePairCommentsSum(X_train);
      const X_train_features = prepareFeatures(X_train, allHashtags, pairCommentsSum);

      const likes_model = new MLR(X_train_features, y_likes_train.map(y => [y]));
      const share_model = new MLR(X_train_features, y_share_train.map(y => [y]));
      const comments_model = new MLR(X_train_features, y_comments_train.map(y => [y]));

      const new_ad_processed = {
        hashtags: new_ad.hashtags,
        hour: moment(new_ad.created_at, moment.ISO_8601).hour(),
        dayofweek: moment(new_ad.created_at, moment.ISO_8601).day(),
        comments: 0
      };

      const new_ad_pairs = getHashtagPairs(new_ad_processed.hashtags);
      let new_ad_totalPairComments = 0;
      new_ad_pairs.forEach(pair => {
        new_ad_totalPairComments += pairCommentsSum[pair] || 0;
      });

      let hourBoost = 1;
      const goldenHours = await calculateGoldenHours(postsData, commentCounts)
      if (goldenHours.includes(new_ad_processed.hour)) {
        hourBoost = 1.15;
      }

      const new_ad_features = [
        ...prepareFeatures([new_ad_processed], allHashtags, pairCommentsSum)[0].slice(0, -1),
        new_ad_totalPairComments,
        hourBoost
      ];

      likesSum += Math.max(0, Math.round(likes_model.predict([new_ad_features])[0][0]));
      sharesSum += Math.max(0, Math.round(share_model.predict([new_ad_features])[0][0]));
      commentsSum += Math.max(0, Math.round(comments_model.predict([new_ad_features])[0][0]));
    }

    console.log(`👍 Likes: ${Math.round(likesSum / runs)}`);
    console.log(`🔁 Shares: ${Math.round(sharesSum / runs)}`);
    console.log(`💬 Comments: ${Math.round(commentsSum / runs)}`);
    
    return {
      likes: likesSum,
      shares: sharesSum,
      comment: commentsSum
    }
  } catch (error) {
    console.error('Lỗi:', error.message);
  }
}




module.exports = { trainAndPredictFromInput };

