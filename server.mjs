import { readFileSync, writeFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ImapFlow } from "imapflow";
import { simpleParser } from "mailparser";
import { z } from "zod";

const __dirname = dirname(fileURLToPath(import.meta.url));
try {
  const envFile = readFileSync(resolve(__dirname, ".env"), "utf8");
  for (const line of envFile.split("\n")) {
    const idx = line.indexOf("=");
    if (idx === -1) continue;
    const key = line.slice(0, idx).trim();
    const val = line.slice(idx + 1).trim();
    if (key && !process.env[key]) process.env[key] = val;
  }
} catch {}

const YAHOO_EMAIL = process.env.YAHOO_EMAIL;
const YAHOO_APP_PASSWORD = process.env.YAHOO_APP_PASSWORD;
const FAMILY_PATTERNS = (process.env.FAMILY_CONTACTS || "chandler")
  .split(",")
  .map((s) => s.trim().toLowerCase())
  .filter(Boolean);
const FAMILY_EXCLUDE_DOMAINS = (process.env.FAMILY_EXCLUDE_DOMAINS || "")
  .split(",")
  .map((s) => s.trim().toLowerCase())
  .filter(Boolean);
const SPOUSE_EMAIL = (process.env.SPOUSE_EMAIL || "").trim().toLowerCase();

function log(msg) {
  process.stderr.write(`[yahoo-mail] ${msg}\n`);
}

log(`Starting with email: ${YAHOO_EMAIL}`);
log(`Password configured: ${YAHOO_APP_PASSWORD ? "yes (" + YAHOO_APP_PASSWORD.length + " chars)" : "NO"}`);

// ── Training data persistence ──

const TRAINING_PATH = resolve(__dirname, "training_data.json");

const STOP_WORDS = new Set([
  "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
  "was", "one", "our", "out", "has", "his", "how", "its", "may", "new",
  "now", "old", "see", "way", "who", "did", "get", "got", "had", "has",
  "him", "let", "say", "she", "too", "use", "with", "this", "that", "from",
  "they", "been", "have", "were", "will", "your", "what", "when", "make",
  "like", "just", "over", "such", "take", "than", "them", "very", "some",
  "into", "most", "also", "after", "about", "more", "here", "there",
  "which", "their", "would", "could", "other", "these", "those", "being",
  "where", "while", "should", "before", "through", "because",
]);

function freshTrainingData() {
  return {
    senders: {},
    domains: {},
    fingerprints: { marketing: [], hard_spam: [] },
    keywords: { marketing: {}, hard_spam: {}, not_spam: {} },
    trained_uids: [],
    total_trained: 0,
    last_trained: null,
  };
}

function loadTrainingData() {
  try {
    if (existsSync(TRAINING_PATH)) {
      const raw = JSON.parse(readFileSync(TRAINING_PATH, "utf8"));
      // Migrate from v1 format
      if (!raw.fingerprints) raw.fingerprints = { marketing: [], hard_spam: [] };
      if (!raw.trained_uids) raw.trained_uids = [];
      if (raw.subject_words && !raw.keywords) {
        raw.keywords = { marketing: {}, hard_spam: {}, not_spam: {} };
        // Don't migrate old polluted keywords — start fresh
        delete raw.subject_words;
      }
      if (!raw.keywords) raw.keywords = { marketing: {}, hard_spam: {}, not_spam: {} };
      return raw;
    }
  } catch (err) {
    log(`Error loading training data: ${err.message}`);
  }
  return freshTrainingData();
}

function saveTrainingData(data) {
  writeFileSync(TRAINING_PATH, JSON.stringify(data, null, 2), "utf8");
  log(`Training data saved (${data.total_trained} classifications, ${data.fingerprints.hard_spam.length + data.fingerprints.marketing.length} fingerprints)`);
}

// ── Text processing ──

function extractDomain(email) {
  const match = (email || "").match(/@(.+)/);
  return match ? match[1].toLowerCase() : null;
}

function isGarbage(token) {
  if (token.length > 25) return true;
  if (token.length < 3) return true;
  if (/^\d+$/.test(token)) return true;
  if (/^[0-9a-f]{8,}$/i.test(token)) return true;
  if (/^[a-z0-9+/=]{20,}$/i.test(token)) return true;
  if (token.includes("http")) return true;
  if (STOP_WORDS.has(token)) return true;
  return false;
}

function extractWords(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z\s'-]/g, " ")
    .split(/\s+/)
    .filter((w) => !isGarbage(w));
}

function normalizeBody(text) {
  return (text || "").toLowerCase().replace(/\s+/g, " ").trim();
}

function extractFingerprints(text) {
  const normalized = normalizeBody(text);
  const phrases = [];

  // Extract sentences or clause-level chunks (split on period, newline, or long gaps)
  const chunks = normalized
    .split(/[.!?\n]|(?:\s{3,})/)
    .map((c) => c.trim())
    .filter((c) => c.length >= 30 && c.length <= 200);

  for (const chunk of chunks) {
    const wordCount = chunk.split(/\s+/).length;
    if (wordCount >= 6) {
      phrases.push(chunk);
    }
  }
  return phrases;
}

function fingerprintOverlap(phrase1, phrase2) {
  const words1 = new Set(phrase1.split(/\s+/));
  const words2 = new Set(phrase2.split(/\s+/));
  let intersection = 0;
  for (const w of words1) {
    if (words2.has(w)) intersection++;
  }
  const union = new Set([...words1, ...words2]).size;
  return union > 0 ? intersection / union : 0;
}

// ── Classifier ──

const HARD_SPAM_PATTERNS = [
  /\bcongratulations?\b.*\bwon\b/i,
  /\bviagra\b/i,
  /\bcialis\b/i,
  /\blottery\b/i,
  /\bnigerian?\b/i,
  /\bprince\b.*\bmillion/i,
  /\bweight.?loss\b/i,
  /\bdiet\b.*\bpill/i,
  /\bcrypto\b.*\binvest/i,
  /\bbitcoin\b.*\bdouble/i,
  /\baccount.*\bsuspend/i,
  /\bverify.*\baccount\b/i,
  /\bpassword.*\bexpir/i,
  /\burgent.*\baction\b/i,
  /\brisk.?free\b/i,
  /\bfree gift\b/i,
];

const MARKETING_PATTERNS = [
  /\bunsubscribe\b/i,
  /\bact now\b/i,
  /\blimited time\b/i,
  /\bno cost\b/i,
  /\b\d+%\s*off\b/i,
  /\bspecial offer\b/i,
  /\bdeal\s*(of|for)\b/i,
  /\bnewsletter\b/i,
  /\bpromo(tion)?\b/i,
  /\bcoupon\b/i,
  /\bfree shipping\b/i,
  /\bexclusive\b.*\boffer\b/i,
  /\bflash sale\b/i,
];

function classifyEmail(envelope, textSnippet, trainingData) {
  const fromAddr = (envelope.from?.[0]?.address || "").toLowerCase();
  const domain = extractDomain(fromAddr);
  const subject = (envelope.subject || "");
  const combined = `${subject} ${textSnippet}`;
  const normalizedBody = normalizeBody(combined);
  const words = extractWords(combined);

  if (isFamilyEmail(envelope)) return { category: "not_spam", confidence: 1, reason: "family contact" };

  const signals = [];
  let categoryVotes = { marketing: 0, hard_spam: 0, not_spam: 0 };

  // Signal 1: Known sender (strong but not absolute)
  if (trainingData.senders[fromAddr]) {
    const cat = trainingData.senders[fromAddr];
    categoryVotes[cat] += 3;
    signals.push(`sender:${cat}`);
  }

  // Signal 2: Known domain
  if (domain && trainingData.domains[domain]) {
    const cat = trainingData.domains[domain];
    categoryVotes[cat] += 2;
    signals.push(`domain:${cat}`);
  }

  // Signal 3: Fingerprint matching (the key generalization mechanism)
  for (const cat of ["hard_spam", "marketing"]) {
    const fps = trainingData.fingerprints[cat] || [];
    let bestMatch = 0;
    let bestPhrase = "";
    for (const fp of fps) {
      const phrase = typeof fp === "string" ? fp : fp.phrase;
      if (!phrase) continue;
      // Substring containment (exact boilerplate match)
      if (normalizedBody.includes(phrase)) {
        bestMatch = Math.max(bestMatch, 1.0);
        bestPhrase = phrase.slice(0, 50);
        break;
      }
      // Fuzzy overlap for near-matches
      const overlap = fingerprintOverlap(phrase, normalizedBody);
      if (overlap > bestMatch) {
        bestMatch = overlap;
        bestPhrase = phrase.slice(0, 50);
      }
    }
    if (bestMatch >= 0.5) {
      const weight = bestMatch >= 0.9 ? 4 : bestMatch >= 0.7 ? 3 : 2;
      categoryVotes[cat] += weight;
      signals.push(`fingerprint:${cat}(${(bestMatch * 100).toFixed(0)}% "${bestPhrase}...")`);
    }
  }

  // Signal 4: Built-in regex patterns
  let hardSpamPatterns = 0;
  for (const pattern of HARD_SPAM_PATTERNS) {
    if (pattern.test(combined)) hardSpamPatterns++;
  }
  if (hardSpamPatterns > 0) {
    categoryVotes.hard_spam += hardSpamPatterns * 1.5;
    signals.push(`patterns:hard_spam(${hardSpamPatterns})`);
  }

  let marketingPatterns = 0;
  for (const pattern of MARKETING_PATTERNS) {
    if (pattern.test(combined)) marketingPatterns++;
  }
  if (marketingPatterns > 0) {
    categoryVotes.marketing += marketingPatterns;
    signals.push(`patterns:marketing(${marketingPatterns})`);
  }

  if (/noreply|no-reply|donotreply/.test(fromAddr)) {
    categoryVotes.marketing += 0.5;
    signals.push("noreply-sender");
  }

  // Signal 5: Trained keyword scoring
  for (const cat of ["marketing", "hard_spam", "not_spam"]) {
    const kw = trainingData.keywords[cat] || {};
    let score = 0;
    let hits = 0;
    for (const word of words) {
      if (kw[word]) {
        score += kw[word];
        hits++;
      }
    }
    if (hits >= 3 && score >= 2) {
      categoryVotes[cat] += Math.min(score * 0.3, 3);
      signals.push(`keywords:${cat}(${hits} hits, ${score.toFixed(1)} score)`);
    }
  }

  // Decision: pick highest-voted category
  const entries = Object.entries(categoryVotes);
  entries.sort((a, b) => b[1] - a[1]);
  const [topCat, topScore] = entries[0];
  const [, secondScore] = entries[1];

  if (topCat === "not_spam" || topScore < 1.5) {
    return {
      category: "not_spam",
      confidence: topCat === "not_spam" ? Math.min(0.9, 0.4 + topScore * 0.1) : 0.4,
      reason: signals.length ? signals.join(" + ") : "no signals",
    };
  }

  // Confidence based on margin between top and second category
  const margin = topScore - secondScore;
  const totalSignals = signals.length;
  const confidence = Math.min(0.95, Math.max(0.4,
    0.4 + margin * 0.08 + totalSignals * 0.05 + topScore * 0.03
  ));

  return {
    category: topCat,
    confidence,
    reason: signals.join(" + "),
  };
}

// ── IMAP helpers ──

const IMAP_CONFIG = {
  host: "imap.mail.yahoo.com",
  port: 993,
  secure: true,
  auth: { user: YAHOO_EMAIL, pass: YAHOO_APP_PASSWORD },
  logger: false,
};

async function getImapClient() {
  log(`Connecting to imap.mail.yahoo.com:993 as ${YAHOO_EMAIL}...`);
  const client = new ImapFlow(IMAP_CONFIG);
  try {
    await client.connect();
    log("IMAP connected successfully");
    return client;
  } catch (err) {
    log(`IMAP connection FAILED: ${err.message}`);
    throw err;
  }
}

function isFamilyEmail(envelope) {
  const addresses = [
    ...(envelope.from || []),
    ...(envelope.sender || []),
    ...(envelope.replyTo || []),
  ];
  const allFields = addresses
    .map((a) => `${a.name || ""} ${a.address || ""}`.toLowerCase())
    .join(" ");
  if (!FAMILY_PATTERNS.some((pattern) => allFields.includes(pattern))) return false;

  const fromAddr = (envelope.from?.[0]?.address || "").toLowerCase();
  const fromDomain = extractDomain(fromAddr);
  if (fromDomain && FAMILY_EXCLUDE_DOMAINS.some((d) => fromDomain === d || fromDomain.endsWith("." + d))) {
    return false;
  }

  if (SPOUSE_EMAIL && fromAddr === SPOUSE_EMAIL) {
    const recipientCount = (envelope.to || []).length + (envelope.cc || []).length;
    if (recipientCount <= 1) return false;
  }

  return true;
}

function formatAddress(addr) {
  if (!addr) return "unknown";
  if (Array.isArray(addr)) {
    return addr.map((a) => (a.name ? `${a.name} <${a.address}>` : a.address)).join(", ");
  }
  return addr.name ? `${addr.name} <${addr.address}>` : addr.address;
}

// ── MCP Server ──

const server = new McpServer({
  name: "yahoo-mail",
  version: "3.0.0",
});

// ── Training tools ──

server.tool(
  "get_training_batch",
  "Fetch a batch of UNTRAINED emails for spam classification. Skips already-classified emails and supports pagination via offset. Returns emails with previews so you can classify each as 'not_spam', 'marketing', or 'hard_spam'.",
  {
    count: z.number().min(5).max(50).default(30).describe("Number of untrained emails to return"),
    mailbox: z.string().default("INBOX").describe("Folder to pull from (try 'Bulk Mail' or 'Junk' for spam samples)"),
    offset: z.number().min(0).default(0).describe("Skip this many messages from the end of the mailbox (for pagination into older mail)"),
  },
  async ({ count, mailbox, offset }) => {
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock(mailbox);
      try {
        const trainingData = loadTrainingData();
        const trainedSet = new Set(trainingData.trained_uids);

        const emails = [];
        const total = client.mailbox.exists;
        // Scan more than needed since we skip trained UIDs
        const scanSize = Math.min(total, (count + trainedSet.size + offset) * 2);
        const startSeq = Math.max(1, total - scanSize - offset + 1);
        const endSeq = total - offset;

        if (endSeq < 1) {
          return { content: [{ type: "text", text: `Offset ${offset} exceeds mailbox size (${total} messages).` }] };
        }

        for await (const msg of client.fetch(`${startSeq}:${endSeq}`, {
          envelope: true,
          flags: true,
          source: true,
        })) {
          if (trainedSet.has(msg.uid)) continue;

          let preview = "";
          if (msg.source) {
            try {
              const parsed = await simpleParser(msg.source);
              preview = (parsed.text || "").slice(0, 200).replace(/\s+/g, " ").trim();
            } catch {
              preview = "(could not parse)";
            }
          }

          emails.push({
            uid: msg.uid,
            from: formatAddress(msg.envelope.from),
            fromAddress: msg.envelope.from?.[0]?.address || "",
            subject: msg.envelope.subject || "(no subject)",
            date: msg.envelope.date?.toISOString(),
            preview,
            isFamily: isFamilyEmail(msg.envelope),
          });
        }

        // Most recent first, cap at requested count
        emails.reverse();
        const batch = emails.slice(0, count);

        if (batch.length === 0) {
          return {
            content: [{
              type: "text",
              text: `No untrained emails found in ${mailbox} (offset: ${offset}, scanned ${scanSize} messages, ${trainedSet.size} already trained).\nTry increasing offset or scanning a different mailbox (e.g., "Bulk Mail", "Junk").`,
            }],
          };
        }

        const list = batch
          .map((e, i) => `${i + 1}. UID:${e.uid}${e.isFamily ? " [FAMILY]" : ""}
   From: ${e.from}
   Subject: ${e.subject}
   Date: ${e.date?.slice(0, 16)}
   Preview: ${e.preview}`)
          .join("\n\n");

        return {
          content: [{
            type: "text",
            text: `Training batch: ${batch.length} untrained emails from ${mailbox} (offset: ${offset})\n${trainedSet.size} previously trained | ${total} total in mailbox\n\nClassify each as: "not_spam" | "marketing" | "hard_spam"\n\n${list}\n\nUse submit_training with [{uid, category}, ...] when ready.`,
          }],
        };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "submit_training",
  "Submit email classifications to train the spam filter. Learns sender patterns, content fingerprints, and keywords. Each entry maps a UID to 'not_spam', 'marketing', or 'hard_spam'.",
  {
    classifications: z
      .array(z.object({
        uid: z.number().describe("Email UID from the training batch"),
        category: z.enum(["not_spam", "marketing", "hard_spam"]).describe("Your classification"),
      }))
      .min(1)
      .describe("Array of {uid, category} classifications"),
    mailbox: z.string().default("INBOX").describe("Folder the training emails came from"),
  },
  async ({ classifications, mailbox }) => {
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock(mailbox);
      try {
        const trainingData = loadTrainingData();
        const uidList = classifications.map((c) => c.uid).join(",");
        const categoryMap = Object.fromEntries(classifications.map((c) => [c.uid, c.category]));

        let processed = 0;
        let skippedFamily = 0;
        const newFingerprints = { marketing: [], hard_spam: [] };

        for await (const msg of client.fetch(uidList, {
          envelope: true,
          source: true,
        }, { uid: true })) {
          const category = categoryMap[msg.uid];
          if (!category) continue;

          if (isFamilyEmail(msg.envelope)) {
            skippedFamily++;
            continue;
          }

          const fromAddr = (msg.envelope.from?.[0]?.address || "").toLowerCase();
          const domain = extractDomain(fromAddr);
          const subject = msg.envelope.subject || "";

          let bodyText = "";
          if (msg.source) {
            try {
              const parsed = await simpleParser(msg.source);
              bodyText = (parsed.text || "").slice(0, 1000);
            } catch {}
          }

          // Learn sender
          if (fromAddr) trainingData.senders[fromAddr] = category;

          // Learn domain (after 2+ senders from same domain share a category)
          if (domain) {
            const sameDomainSenders = Object.entries(trainingData.senders)
              .filter(([addr, cat]) => extractDomain(addr) === domain && cat === category);
            if (sameDomainSenders.length >= 2) {
              trainingData.domains[domain] = category;
            }
          }

          // Extract and store fingerprints for spam/marketing
          if (category !== "not_spam") {
            const phrases = extractFingerprints(bodyText);
            for (const phrase of phrases) {
              newFingerprints[category].push(phrase);
            }
          }

          // Learn clean keywords
          const words = extractWords(`${subject} ${bodyText}`);
          const kw = trainingData.keywords[category];
          for (const word of words) {
            kw[word] = (kw[word] || 0) + 1;
          }
          // Decay opposing categories
          for (const otherCat of ["not_spam", "marketing", "hard_spam"]) {
            if (otherCat === category) continue;
            const otherKw = trainingData.keywords[otherCat];
            for (const word of words) {
              if (otherKw[word]) {
                otherKw[word] = Math.max(0, otherKw[word] - 0.3);
                if (otherKw[word] === 0) delete otherKw[word];
              }
            }
          }

          // Track this UID as trained
          trainingData.trained_uids.push(msg.uid);
          processed++;
        }

        // Deduplicate fingerprints: only add phrases not already stored
        for (const cat of ["marketing", "hard_spam"]) {
          const existing = new Set(trainingData.fingerprints[cat].map((fp) =>
            typeof fp === "string" ? fp : fp.phrase
          ));
          // Normalize existing to plain strings
          trainingData.fingerprints[cat] = trainingData.fingerprints[cat].map((fp) =>
            typeof fp === "string" ? fp : fp.phrase
          );

          for (const phrase of newFingerprints[cat]) {
            // Check if substantially similar to an existing fingerprint
            let dominated = false;
            for (const ex of existing) {
              if (fingerprintOverlap(phrase, ex) > 0.7) {
                dominated = true;
                break;
              }
            }
            if (!dominated) {
              trainingData.fingerprints[cat].push(phrase);
              existing.add(phrase);
            }
          }
        }

        // Cap trained_uids to prevent unbounded growth
        if (trainingData.trained_uids.length > 5000) {
          trainingData.trained_uids = trainingData.trained_uids.slice(-5000);
        }

        trainingData.total_trained += processed;
        trainingData.last_trained = new Date().toISOString();
        saveTrainingData(trainingData);

        const senderCounts = { not_spam: 0, marketing: 0, hard_spam: 0 };
        for (const cat of Object.values(trainingData.senders)) {
          senderCounts[cat] = (senderCounts[cat] || 0) + 1;
        }

        const fpCount = trainingData.fingerprints.hard_spam.length + trainingData.fingerprints.marketing.length;
        const newFpCount = newFingerprints.hard_spam.length + newFingerprints.marketing.length;

        return {
          content: [{
            type: "text",
            text: `Training updated!\n\nThis session: ${processed} classified${skippedFamily ? ` (${skippedFamily} family skipped)` : ""}\nNew fingerprints extracted: ${newFpCount}\n\nTotals:\n  ${trainingData.total_trained} classifications\n  ${Object.keys(trainingData.senders).length} senders (${senderCounts.not_spam} safe, ${senderCounts.marketing} marketing, ${senderCounts.hard_spam} spam)\n  ${Object.keys(trainingData.domains).length} domains\n  ${fpCount} content fingerprints\n  ${trainingData.trained_uids.length} UIDs tracked\n  Last trained: ${trainingData.last_trained}`,
          }],
        };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "training_status",
  "Check the spam filter's training state: senders, domains, fingerprints, and keyword statistics.",
  {},
  async () => {
    const trainingData = loadTrainingData();

    const sendersByCategory = { not_spam: [], marketing: [], hard_spam: [] };
    for (const [sender, cat] of Object.entries(trainingData.senders)) {
      sendersByCategory[cat]?.push(sender);
    }
    const domainsByCategory = { not_spam: [], marketing: [], hard_spam: [] };
    for (const [domain, cat] of Object.entries(trainingData.domains)) {
      domainsByCategory[cat]?.push(domain);
    }

    const topWords = (wordMap, n = 10) =>
      Object.entries(wordMap)
        .sort((a, b) => b[1] - a[1])
        .slice(0, n)
        .map(([w, s]) => `${w}(${s.toFixed(1)})`)
        .join(", ");

    const fpSummary = (cat) => {
      const fps = trainingData.fingerprints[cat] || [];
      if (fps.length === 0) return "  none";
      return fps.map((fp) => `  "${(typeof fp === "string" ? fp : fp.phrase).slice(0, 70)}..."`).join("\n");
    };

    return {
      content: [{
        type: "text",
        text: `Spam Filter Training Status\n${"=".repeat(40)}\n\nTotal classifications: ${trainingData.total_trained}\nTrained UIDs tracked: ${trainingData.trained_uids.length}\nLast trained: ${trainingData.last_trained || "never"}\n\nKnown senders:\n  Safe: ${sendersByCategory.not_spam.length}\n  Marketing: ${sendersByCategory.marketing.length}\n  Hard spam: ${sendersByCategory.hard_spam.length}\n\nKnown domains:\n  Safe: ${domainsByCategory.not_spam.join(", ") || "none"}\n  Marketing: ${domainsByCategory.marketing.join(", ") || "none"}\n  Hard spam: ${domainsByCategory.hard_spam.join(", ") || "none"}\n\nContent fingerprints:\n  Hard spam (${(trainingData.fingerprints.hard_spam || []).length}):\n${fpSummary("hard_spam")}\n  Marketing (${(trainingData.fingerprints.marketing || []).length}):\n${fpSummary("marketing")}\n\nTop keywords:\n  Marketing: ${topWords(trainingData.keywords.marketing) || "none"}\n  Hard spam: ${topWords(trainingData.keywords.hard_spam) || "none"}\n  Safe: ${topWords(trainingData.keywords.not_spam) || "none"}\n\nTo add more training data, use get_training_batch.`,
      }],
    };
  }
);

server.tool(
  "reset_training",
  "Reset all training data and start fresh.",
  { confirm: z.literal(true).describe("Must be true to confirm reset") },
  async ({ confirm }) => {
    if (!confirm) {
      return { content: [{ type: "text", text: "Reset cancelled." }] };
    }
    saveTrainingData(freshTrainingData());
    return { content: [{ type: "text", text: "Training data reset. Use get_training_batch to start fresh." }] };
  }
);

// ── Spam scanning ──

server.tool(
  "scan_spam",
  "Scan inbox for spam using trained classifier + fingerprints + patterns. Returns two lists: hard_spam and marketing. Does NOT move anything.",
  {
    count: z.number().min(1).max(100).default(30).describe("Number of recent emails to scan"),
    min_confidence: z.number().min(0).max(1).default(0.5).describe("Minimum confidence to flag (0=aggressive, 0.8=conservative)"),
  },
  async ({ count, min_confidence }) => {
    const trainingData = loadTrainingData();
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock("INBOX");
      try {
        const marketing = [];
        const hardSpam = [];
        const total = client.mailbox.exists;
        const startSeq = Math.max(1, total - count + 1);

        for await (const msg of client.fetch(`${startSeq}:*`, {
          envelope: true,
          source: true,
        })) {
          let textSnippet = "";
          if (msg.source) {
            try {
              const parsed = await simpleParser(msg.source);
              textSnippet = (parsed.text || "").slice(0, 1000);
            } catch {}
          }

          const result = classifyEmail(msg.envelope, textSnippet, trainingData);
          if (result.category === "not_spam") continue;
          if (result.confidence < min_confidence) continue;

          const entry = {
            uid: msg.uid,
            subject: msg.envelope.subject || "(no subject)",
            from: formatAddress(msg.envelope.from),
            date: msg.envelope.date?.toISOString(),
            confidence: result.confidence,
            reason: result.reason,
          };

          if (result.category === "marketing") marketing.push(entry);
          else if (result.category === "hard_spam") hardSpam.push(entry);
        }

        marketing.sort((a, b) => b.confidence - a.confidence);
        hardSpam.sort((a, b) => b.confidence - a.confidence);

        if (marketing.length === 0 && hardSpam.length === 0) {
          return {
            content: [{
              type: "text",
              text: `No spam found in last ${count} messages (min confidence: ${min_confidence}).\nTraining: ${trainingData.total_trained} classifications, ${(trainingData.fingerprints.hard_spam?.length || 0) + (trainingData.fingerprints.marketing?.length || 0)} fingerprints.${trainingData.total_trained === 0 ? "\nTip: Use get_training_batch to train the classifier." : ""}`,
            }],
          };
        }

        let output = `Scan: ${count} emails checked (confidence >= ${(min_confidence * 100).toFixed(0)}%):\n`;

        if (hardSpam.length > 0) {
          output += `\nHARD SPAM (${hardSpam.length}):\n`;
          output += hardSpam
            .map((s) => `  UID:${s.uid} [${(s.confidence * 100).toFixed(0)}%] ${s.from} | ${s.subject}\n    ${s.reason}`)
            .join("\n");
        }

        if (marketing.length > 0) {
          output += `\n\nMARKETING (${marketing.length}):\n`;
          output += marketing
            .map((s) => `  UID:${s.uid} [${(s.confidence * 100).toFixed(0)}%] ${s.from} | ${s.subject}\n    ${s.reason}`)
            .join("\n");
        }

        output += "\n\nActions:";
        if (hardSpam.length) output += `\n  move_to_junk UIDs: [${hardSpam.map((s) => s.uid).join(", ")}]`;
        if (marketing.length) output += `\n  move_to_folder UIDs: [${marketing.map((s) => s.uid).join(", ")}]`;

        return { content: [{ type: "text", text: output }] };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

// ── Email movement tools ──

server.tool(
  "move_to_junk",
  "Move hard spam emails to Bulk Mail by UID.",
  { uids: z.array(z.number()).min(1).describe("Email UIDs to junk") },
  async ({ uids }) => {
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock("INBOX");
      try {
        await client.messageMove(uids.join(","), "Bulk Mail", { uid: true });
        return { content: [{ type: "text", text: `Moved ${uids.length} message(s) to Bulk Mail.\nUIDs: ${uids.join(", ")}` }] };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "move_to_folder",
  "Move emails to a specified folder by UID. Creates the folder if needed.",
  {
    uids: z.array(z.number()).min(1).describe("Email UIDs to move"),
    folder: z.string().default("Marketing").describe("Destination folder"),
  },
  async ({ uids, folder }) => {
    const client = await getImapClient();
    try {
      try { await client.mailboxCreate(folder); } catch {}
      const lock = await client.getMailboxLock("INBOX");
      try {
        await client.messageMove(uids.join(","), folder, { uid: true });
        return { content: [{ type: "text", text: `Moved ${uids.length} message(s) to "${folder}".\nUIDs: ${uids.join(", ")}` }] };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "move_to_inbox",
  "Rescue emails from Junk or Marketing back to Inbox.",
  {
    uids: z.array(z.number()).min(1).describe("Email UIDs to move back to Inbox"),
    from_folder: z.string().default("Bulk Mail").describe("Source folder"),
  },
  async ({ uids, from_folder }) => {
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock(from_folder);
      try {
        await client.messageMove(uids.join(","), "INBOX", { uid: true });
        return { content: [{ type: "text", text: `Rescued ${uids.length} message(s) from "${from_folder}" to Inbox.` }] };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "sweep_sender",
  "Bulk-move ALL emails from a sender pattern to a destination folder (or Bulk Mail). Loops internally until no matches remain, so one call fully drains a sender. Returns total moved.",
  {
    sender: z.string().describe("Sender pattern to match (e.g. 'nextdoor.com', 'krispykreme.com', 'dccc.org')"),
    destination: z.enum(["Marketing", "Purge-Candidates", "Bulk Mail", "Receipts"]).default("Marketing").describe("Destination folder"),
  },
  async ({ sender, destination }) => {
    const { totalMoved, passes } = await runSweep(sender, destination);
    return {
      content: [{
        type: "text",
        text: totalMoved === 0
          ? `No emails found from "${sender}" in INBOX.`
          : `Swept ${totalMoved} email(s) from "${sender}" to "${destination}" in ${passes} pass(es).`,
      }],
    };
  }
);

const SWEEP_RULES_PATH = resolve(__dirname, "sweep_rules.json");

async function runSweep(sender, destination, mailbox = "INBOX", batch_size = 100) {
  let totalMoved = 0;
  let passes = 0;
  const maxPasses = 50;

  while (passes < maxPasses) {
    passes++;
    const client = await getImapClient();
    let batchMoved = 0;
    try {
      const lock = await client.getMailboxLock(mailbox);
      try {
        const uids = await client.search({ from: sender }, { uid: true });
        if (uids.length === 0) break;

        const batch = uids.slice(-batch_size);
        if (destination !== "Bulk Mail") {
          try { await client.mailboxCreate(destination); } catch {}
        }
        await client.messageMove(batch.join(","), destination, { uid: true });
        batchMoved = batch.length;
        totalMoved += batchMoved;
        log(`sweep: moved ${batchMoved} from "${sender}" → "${destination}" (${totalMoved} total, pass ${passes})`);
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
    if (batchMoved === 0) break;
  }
  return { totalMoved, passes };
}

server.tool(
  "sweep_all",
  "Run ALL rules from sweep_rules.json against the inbox. Each sender is fully drained before moving to the next. Returns a summary of everything moved.",
  {
    dry_run: z.boolean().default(false).describe("If true, just report what would be swept without moving anything"),
  },
  async ({ dry_run }) => {
    let rules;
    try {
      rules = JSON.parse(readFileSync(SWEEP_RULES_PATH, "utf8")).rules;
    } catch (err) {
      return { content: [{ type: "text", text: `Error reading sweep_rules.json: ${err.message}` }] };
    }

    if (dry_run) {
      const client = await getImapClient();
      const counts = [];
      try {
        const lock = await client.getMailboxLock("INBOX");
        try {
          for (const rule of rules) {
            const uids = await client.search({ from: rule.sender }, { uid: true });
            if (uids.length > 0) {
              counts.push(`${rule.sender} → ${rule.destination}: ${uids.length} email(s) [${rule.note}]`);
            }
          }
        } finally {
          lock.release();
        }
      } finally {
        await client.logout();
      }
      return {
        content: [{ type: "text", text: counts.length === 0
          ? "No matching emails found for any rules."
          : `Dry run — would sweep:\n\n${counts.join("\n")}\n\nTotal rules with matches: ${counts.length}` }],
      };
    }

    const results = [];
    let grandTotal = 0;
    for (const rule of rules) {
      const { totalMoved, passes } = await runSweep(rule.sender, rule.destination);
      if (totalMoved > 0) {
        results.push(`${rule.sender} → ${rule.destination}: ${totalMoved} moved (${passes} passes) [${rule.note}]`);
        grandTotal += totalMoved;
      }
    }

    return {
      content: [{
        type: "text",
        text: grandTotal === 0
          ? "No emails matched any sweep rules. Inbox is clean!"
          : `Sweep complete: ${grandTotal} total email(s) moved across ${results.length} sender(s).\n\n${results.join("\n")}`,
      }],
    };
  }
);

// ── Inbox and search tools ──

server.tool(
  "check_inbox",
  "Check Yahoo Mail inbox. Returns a summary of the latest messages with family and spam indicators.",
  { count: z.number().min(1).max(50).default(10).describe("Number of recent emails to fetch") },
  async ({ count }) => {
    const trainingData = loadTrainingData();
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock("INBOX");
      try {
        const messages = [];
        const total = client.mailbox.exists;
        const startSeq = Math.max(1, total - count + 1);

        for await (const msg of client.fetch(`${startSeq}:*`, {
          envelope: true,
          flags: true,
          source: true,
        })) {
          let textSnippet = "";
          if (msg.source) {
            try {
              const parsed = await simpleParser(msg.source);
              textSnippet = (parsed.text || "").slice(0, 500);
            } catch {}
          }

          const classification = classifyEmail(msg.envelope, textSnippet, trainingData);

          messages.push({
            uid: msg.uid,
            subject: msg.envelope.subject || "(no subject)",
            from: formatAddress(msg.envelope.from),
            date: msg.envelope.date?.toISOString(),
            flags: [...(msg.flags || [])],
            isFamily: isFamilyEmail(msg.envelope),
            spamCategory: classification.category,
            spamConfidence: classification.confidence,
          });
        }

        messages.reverse();
        const summary = messages
          .map((m) => {
            let tag = "";
            if (m.isFamily) tag = "[FAMILY] ";
            else if (m.spamCategory === "hard_spam" && m.spamConfidence >= 0.5) tag = "[SPAM] ";
            else if (m.spamCategory === "marketing" && m.spamConfidence >= 0.5) tag = "[MKTG] ";
            const unread = m.flags.includes("\\Seen") ? "" : " [UNREAD]";
            return `${tag}UID:${m.uid} | ${m.date?.slice(0, 16)} | From: ${m.from} | Subject: ${m.subject}${unread}`;
          })
          .join("\n");

        const familyCount = messages.filter((m) => m.isFamily).length;
        const unreadCount = messages.filter((m) => !m.flags.includes("\\Seen")).length;
        const spamCount = messages.filter((m) => m.spamCategory !== "not_spam" && m.spamConfidence >= 0.5).length;

        return {
          content: [{
            type: "text",
            text: `Inbox: ${total} total | latest ${messages.length} | ${unreadCount} unread | ${familyCount} family | ${spamCount} suspected spam/marketing\n\n${summary}`,
          }],
        };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "summarize_inbox",
  "Fetch recent emails with body previews for summarization. Returns sender, subject, date, and a text preview of each message so the assistant can summarize inbox contents, identify themes, and surface action items.",
  {
    count: z.number().min(1).max(150).default(15).describe("Number of recent emails to fetch"),
    mailbox: z.string().default("INBOX").describe("Folder to summarize"),
    unread_only: z.boolean().default(false).describe("Only include unread emails"),
    preview_length: z.number().min(50).max(1000).default(300).describe("Characters of body text to include per email"),
  },
  async ({ count, mailbox, unread_only, preview_length }) => {
    const trainingData = loadTrainingData();
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock(mailbox);
      try {
        const total = client.mailbox.exists;
        const startSeq = Math.max(1, total - count + 1);
        const messages = [];

        for await (const msg of client.fetch(`${startSeq}:*`, {
          envelope: true,
          flags: true,
          source: true,
        })) {
          const unread = !msg.flags?.has("\\Seen");
          if (unread_only && !unread) continue;

          let preview = "";
          let textSnippet = "";
          if (msg.source) {
            try {
              const parsed = await simpleParser(msg.source);
              const body = parsed.text || parsed.html?.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ") || "";
              preview = body.slice(0, preview_length).replace(/\s+/g, " ").trim();
              textSnippet = body.slice(0, 500);
            } catch {
              preview = "(could not parse)";
            }
          }

          const classification = classifyEmail(msg.envelope, textSnippet, trainingData);

          messages.push({
            uid: msg.uid,
            subject: msg.envelope.subject || "(no subject)",
            from: formatAddress(msg.envelope.from),
            date: msg.envelope.date?.toISOString(),
            unread,
            isFamily: isFamilyEmail(msg.envelope),
            spamCategory: classification.category,
            preview,
          });
        }

        messages.reverse();

        if (messages.length === 0) {
          return {
            content: [{
              type: "text",
              text: `No ${unread_only ? "unread " : ""}emails found in ${mailbox}.`,
            }],
          };
        }

        const list = messages
          .map((m, i) => {
            let tag = "";
            if (m.isFamily) tag = "[FAMILY] ";
            else if (m.spamCategory === "hard_spam") tag = "[SPAM] ";
            else if (m.spamCategory === "marketing") tag = "[MKTG] ";
            const unreadTag = m.unread ? "[UNREAD] " : "";
            return `${i + 1}. ${tag}${unreadTag}${m.subject}\n   From: ${m.from}\n   Date: ${m.date?.slice(0, 16)}\n   Preview: ${m.preview}`;
          })
          .join("\n\n");

        const unreadCount = messages.filter((m) => m.unread).length;
        const familyCount = messages.filter((m) => m.isFamily).length;

        return {
          content: [{
            type: "text",
            text: `${mailbox}: ${messages.length} email(s) | ${unreadCount} unread | ${familyCount} family | ${total} total in folder\n\n${list}`,
          }],
        };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "get_family_emails",
  "Get recent emails from the Chandler family. Searches both INBOX and the !!ChandlerFamily folder.",
  {
    count: z.number().min(1).max(150).default(20).describe("Number of recent emails to scan"),
    unread_only: z.boolean().default(false).describe("Only show unread family emails"),
  },
  async ({ count, unread_only }) => {
    const client = await getImapClient();
    try {
      const familyMessages = [];
      const foldersSearched = [];

      for (const mailbox of ["INBOX", "!!ChandlerFamily"]) {
        let lock;
        try {
          lock = await client.getMailboxLock(mailbox);
        } catch (err) {
          log(`Could not open mailbox "${mailbox}": ${err.message}`);
          continue;
        }
        try {
          foldersSearched.push(mailbox);
          const total = client.mailbox.exists;
          if (total === 0) continue;

          // Use IMAP SEARCH to find family emails by sender pattern instead of scanning sequentially
          let uidsToFetch = [];
          try {
            for (const pattern of FAMILY_PATTERNS) {
              const uids = await client.search({ from: pattern }, { uid: true });
              uidsToFetch.push(...uids);
            }
            // Deduplicate and take most recent
            uidsToFetch = [...new Set(uidsToFetch)].sort((a, b) => b - a).slice(0, count * 2);
          } catch (err) {
            log(`SEARCH failed in ${mailbox}, falling back to sequential scan: ${err.message}`);
            const scanCount = Math.min(total, count * 5);
            const startSeq = Math.max(1, total - scanCount + 1);
            uidsToFetch = null; // signal to use sequence-based fetch
            // Fall through to sequence-based fetch below
          }

          const fetchRange = uidsToFetch && uidsToFetch.length > 0
            ? uidsToFetch.join(",")
            : uidsToFetch === null
              ? `${Math.max(1, total - count * 5 + 1)}:*`
              : null;

          if (!fetchRange) continue;

          const fetchOpts = uidsToFetch ? { uid: true } : {};
          for await (const msg of client.fetch(fetchRange, {
            envelope: true,
            flags: true,
            source: true,
          }, fetchOpts)) {
            if (!isFamilyEmail(msg.envelope)) continue;
            if (unread_only && msg.flags?.has("\\Seen")) continue;

            let preview = "";
            if (msg.source) {
              try {
                const parsed = await simpleParser(msg.source);
                preview = (parsed.text || "").slice(0, 300).replace(/\s+/g, " ").trim();
              } catch {
                preview = "(could not parse body)";
              }
            }

            familyMessages.push({
              uid: msg.uid,
              mailbox,
              subject: msg.envelope.subject || "(no subject)",
              from: formatAddress(msg.envelope.from),
              date: msg.envelope.date?.toISOString(),
              unread: !msg.flags?.has("\\Seen"),
              preview,
            });
          }
        } finally {
          lock.release();
        }
      }

      // Sort all results by date descending, then cap at requested count
      familyMessages.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
      const results = familyMessages.slice(0, count);

      if (results.length === 0) {
        return { content: [{ type: "text", text: `No ${unread_only ? "unread " : ""}Chandler family emails found.\nSearched: ${foldersSearched.join(", ")}\nFamily patterns: ${FAMILY_PATTERNS.join(", ")}` }] };
      }

      const details = results
        .map((m) => `${m.unread ? "[NEW] " : ""}UID:${m.uid} [${m.mailbox}]\nFrom: ${m.from}\nDate: ${m.date?.slice(0, 16)}\nSubject: ${m.subject}\nPreview: ${m.preview}\n`)
        .join("\n---\n");

      return { content: [{ type: "text", text: `Found ${results.length} Chandler family email(s) across ${foldersSearched.join(", ")}:\n\n${details}` }] };
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "read_email",
  "Read the full content of a specific email by UID.",
  {
    uid: z.number().describe("The UID of the email to read"),
    mailbox: z.string().default("INBOX").describe("Folder to read from"),
  },
  async ({ uid, mailbox }) => {
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock(mailbox);
      try {
        const msg = await client.fetchOne(String(uid), { source: true, envelope: true, flags: true }, { uid: true });
        if (!msg) {
          return { content: [{ type: "text", text: `Email UID ${uid} not found in ${mailbox}.` }] };
        }
        const parsed = await simpleParser(msg.source);
        const body = parsed.text || parsed.html?.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ") || "(empty body)";
        return {
          content: [{
            type: "text",
            text: `From: ${formatAddress(msg.envelope.from)}\nTo: ${formatAddress(msg.envelope.to)}\nDate: ${msg.envelope.date?.toISOString()}\nSubject: ${msg.envelope.subject}\n\n${body}`,
          }],
        };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "search_emails",
  "Search emails by criteria (from, subject, date range, etc.)",
  {
    mailbox: z.string().default("INBOX").describe("Folder to search"),
    from: z.string().optional().describe("Search by sender"),
    subject: z.string().optional().describe("Search by subject"),
    since: z.string().optional().describe("Emails since YYYY-MM-DD"),
    before: z.string().optional().describe("Emails before YYYY-MM-DD"),
    unseen: z.boolean().default(false).describe("Only unread"),
    limit: z.number().min(1).max(50).default(20).describe("Max results"),
  },
  async ({ mailbox, from, subject, since, before, unseen, limit }) => {
    const client = await getImapClient();
    try {
      const lock = await client.getMailboxLock(mailbox);
      try {
        const query = {};
        if (from) query.from = from;
        if (subject) query.subject = subject;
        if (since) query.since = new Date(since);
        if (before) query.before = new Date(before);
        if (unseen) query.seen = false;

        const uids = await client.search(query, { uid: true });
        const recentUids = uids.slice(-limit);

        if (recentUids.length === 0) {
          return { content: [{ type: "text", text: "No emails matched." }] };
        }

        const results = [];
        for await (const msg of client.fetch(recentUids.join(","), { envelope: true, flags: true }, { uid: true })) {
          results.push({
            uid: msg.uid,
            subject: msg.envelope.subject || "(no subject)",
            from: formatAddress(msg.envelope.from),
            date: msg.envelope.date?.toISOString(),
            unread: !msg.flags?.has("\\Seen"),
            isFamily: isFamilyEmail(msg.envelope),
          });
        }

        results.reverse();
        const list = results
          .map((r) => `${r.isFamily ? "[FAMILY] " : ""}${r.unread ? "[NEW] " : ""}UID:${r.uid} | ${r.date?.slice(0, 16)} | From: ${r.from} | Subject: ${r.subject}`)
          .join("\n");

        return { content: [{ type: "text", text: `Found ${results.length} email(s) in ${mailbox}:\n\n${list}` }] };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

server.tool(
  "list_folders",
  "List all folders.",
  {},
  async () => {
    const client = await getImapClient();
    try {
      const mailboxes = await client.list();
      const formatted = mailboxes
        .map((mb) => `${mb.path} ${mb.flags ? `[${[...mb.flags].join(", ")}]` : ""}`)
        .join("\n");
      return { content: [{ type: "text", text: `Folders:\n\n${formatted}` }] };
    } finally {
      await client.logout();
    }
  }
);

// ── Receipt summarization ──

const RECEIPT_SUBJECT_KEYWORDS = [
  "receipt", "order confirmation", "payment confirmation",
  "purchase confirmation", "order confirmed", "your order",
  "invoice", "payment received", "shipping confirmation",
  "payment processed", "bill is ready", "thanks for your payment",
];

function extractMerchant(envelope) {
  const fromName = envelope.from?.[0]?.name || "";
  const fromAddr = envelope.from?.[0]?.address || "";

  if (fromName && !/^(no-?reply|mail|info|notification)/i.test(fromName)) {
    const cleaned = fromName.replace(/\s*(support|billing|orders|no-?reply|notifications?|customer\s*service)\s*/gi, "").trim();
    if (cleaned) return cleaned;
  }

  const domain = extractDomain(fromAddr);
  if (domain) {
    const parts = domain.split(".");
    const name = parts.length >= 2 ? parts[parts.length - 2] : parts[0];
    return name.charAt(0).toUpperCase() + name.slice(1);
  }
  return "Unknown";
}

function extractReceiptAmount(text) {
  if (!text) return null;

  const totalPatterns = [
    /(?:grand\s+)?total\s*(?:charged|paid|due|amount)?[\s:]*\$\s*([\d,]+\.?\d{0,2})/i,
    /(?:amount\s+(?:charged|paid|due))[\s:]*\$\s*([\d,]+\.?\d{0,2})/i,
    /(?:you\s+(?:paid|were\s+charged))[\s:]*\$\s*([\d,]+\.?\d{0,2})/i,
    /(?:charge|payment)\s+of\s+\$\s*([\d,]+\.?\d{0,2})/i,
  ];

  for (const pattern of totalPatterns) {
    const match = text.match(pattern);
    if (match) {
      const val = parseFloat(match[1].replace(/,/g, ""));
      if (val > 0 && val < 100000) return val;
    }
  }

  const allAmounts = [...text.matchAll(/\$([\d,]+\.\d{2})/g)]
    .map((m) => parseFloat(m[1].replace(/,/g, "")))
    .filter((n) => n > 0 && n < 100000);

  if (allAmounts.length > 0) return Math.max(...allAmounts);
  return null;
}

function extractOrderNumber(text, subject) {
  const combined = `${subject} ${text}`;
  const patterns = [
    /(?:order|confirmation|transaction|reference|tracking)\s*(?:#|number|no\.?|id)?[\s:#]*([A-Z0-9][\w-]{4,30})/i,
    /#\s*([A-Z0-9][\w-]{4,30})/i,
  ];
  for (const pattern of patterns) {
    const match = combined.match(pattern);
    if (match) return match[1];
  }
  return null;
}

function isLikelyReceipt(subject, text) {
  let score = 0;
  const subjectLower = (subject || "").toLowerCase();
  const combined = `${subjectLower} ${(text || "").toLowerCase()}`;

  if (/receipt/i.test(subjectLower)) score += 3;
  if (/order\s*(confirm|#|\d)/i.test(subjectLower)) score += 3;
  if (/payment\s*(confirm|receipt)/i.test(subjectLower)) score += 3;
  if (/invoice/i.test(subjectLower)) score += 2;
  if (/purchase/i.test(subjectLower)) score += 2;
  if (/shipping\s*confirm/i.test(subjectLower)) score += 1;

  if (/\$[\d,]+\.\d{2}/.test(text)) score += 2;
  if (/total[\s:]/i.test(combined)) score += 1;
  if (/order\s*(number|#|id)/i.test(combined)) score += 1;
  if (/item|qty|quantity|product/i.test(combined)) score += 1;
  if (/subtotal/i.test(combined)) score += 1;
  if (/\btax\b/i.test(combined)) score += 0.5;

  if (/\bunsubscribe\b/i.test(combined)) score -= 1;
  if (/\d+%\s*off/i.test(combined)) score -= 2;
  if (/coupon|promo\s*code/i.test(combined)) score -= 1.5;
  if (/act\s*now|limited\s*time/i.test(combined)) score -= 2;

  return score >= 3;
}

server.tool(
  "summarize_receipts",
  "Scan inbox for receipt and order confirmation emails, extract purchase details (merchant, amount, date, order number), return a formatted summary, and optionally move them to !Receipts.",
  {
    count: z.number().min(1).max(200).default(50).describe("Number of recent emails to scan for receipts"),
    since: z.string().optional().describe("Only include emails since YYYY-MM-DD"),
    move: z.boolean().default(true).describe("Move found receipts to !Receipts folder (false = dry run)"),
  },
  async ({ count, since, move }) => {
    const client = await getImapClient();
    try {
      if (move) {
        try { await client.mailboxCreate("!Receipts"); } catch {}
      }

      const lock = await client.getMailboxLock("INBOX");
      try {
        const candidateUids = new Set();

        // Phase 1a: IMAP SEARCH by subject keywords
        for (const keyword of RECEIPT_SUBJECT_KEYWORDS) {
          try {
            const query = { subject: keyword };
            if (since) query.since = new Date(since);
            const results = await client.search(query, { uid: true });
            for (const uid of results) candidateUids.add(uid);
          } catch {}
        }

        // Phase 1b: Sequential scan fallback
        const total = client.mailbox.exists;
        const startSeq = Math.max(1, total - count + 1);
        for await (const msg of client.fetch(`${startSeq}:*`, { envelope: true })) {
          const subj = (msg.envelope.subject || "").toLowerCase();
          if (RECEIPT_SUBJECT_KEYWORDS.some((kw) => subj.includes(kw))) {
            candidateUids.add(msg.uid);
          }
        }

        if (candidateUids.size === 0) {
          return {
            content: [{
              type: "text",
              text: `No receipt emails found in the last ${count} messages${since ? ` since ${since}` : ""}.\nTip: Try increasing count or adjusting the date range.`,
            }],
          };
        }

        // Phase 2: Fetch, parse, extract
        const receipts = [];
        const uidList = [...candidateUids].sort((a, b) => b - a).slice(0, count);

        for await (const msg of client.fetch(uidList.join(","), {
          source: true,
          envelope: true,
        }, { uid: true })) {
          let text = "";
          try {
            const parsed = await simpleParser(msg.source);
            text = (parsed.text || parsed.html?.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ") || "").slice(0, 3000);
          } catch { continue; }

          const subject = msg.envelope.subject || "";
          if (!isLikelyReceipt(subject, text)) continue;

          receipts.push({
            uid: msg.uid,
            merchant: extractMerchant(msg.envelope),
            amount: extractReceiptAmount(text),
            date: msg.envelope.date?.toISOString()?.slice(0, 10) || "unknown",
            orderNumber: extractOrderNumber(text, subject),
            subject,
          });
        }

        receipts.sort((a, b) => (b.date || "").localeCompare(a.date || ""));

        if (receipts.length === 0) {
          return {
            content: [{
              type: "text",
              text: `Scanned ${candidateUids.size} candidate email(s) but none confirmed as receipts.\nTip: Try increasing count or broadening the date range.`,
            }],
          };
        }

        // Build summary
        let output = `Receipt Summary: ${receipts.length} receipt(s) found`;
        if (since) output += ` since ${since}`;
        output += `\n${"=".repeat(45)}\n\n`;

        receipts.forEach((r, i) => {
          output += `${i + 1}. ${r.merchant}`;
          output += r.amount != null ? ` - $${r.amount.toFixed(2)}` : ` - (amount not found)`;
          output += `\n   Date: ${r.date}`;
          if (r.orderNumber) output += `\n   Order: ${r.orderNumber}`;
          output += `\n   Subject: ${r.subject}\n\n`;
        });

        output += `${"-".repeat(45)}`;
        const receiptsWithAmount = receipts.filter((r) => r.amount != null);
        if (receiptsWithAmount.length > 0) {
          const totalAmount = receiptsWithAmount.reduce((sum, r) => sum + r.amount, 0);
          output += `\nTotal: $${totalAmount.toFixed(2)} across ${receiptsWithAmount.length} receipt(s)`;
        }
        const dates = receipts.map((r) => r.date).filter((d) => d !== "unknown").sort();
        if (dates.length > 0) {
          output += `\nDate range: ${dates[0]} to ${dates[dates.length - 1]}`;
        }

        // Move
        const receiptUids = receipts.map((r) => r.uid);
        if (move) {
          try {
            await client.messageMove(receiptUids.join(","), "!Receipts", { uid: true });
            output += `\nMoved ${receiptUids.length} email(s) to "!Receipts"`;
          } catch (err) {
            output += `\nFailed to move emails: ${err.message}`;
            log(`Move to !Receipts failed: ${err.message}`);
          }
        } else {
          output += `\nDry run - no emails moved (set move=true to organize)`;
          output += `\nUIDs: [${receiptUids.join(", ")}]`;
        }

        return { content: [{ type: "text", text: output }] };
      } finally {
        lock.release();
      }
    } finally {
      await client.logout();
    }
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
