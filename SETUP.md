# Yahoo Mail MCP Server Setup

## 1. Generate a Yahoo App Password

Yahoo requires an app-specific password (not your regular password) for IMAP access.

1. Go to https://login.yahoo.com/myaccount/security/
2. Make sure **Two-step verification** is turned ON
3. Scroll to **Generate and manage app passwords**
4. Select **Other App** and name it "Claude MCP"
5. Copy the generated 16-character password

## 2. Configure the .env file

Edit `.env` in this directory:

```
YAHOO_EMAIL=cchandler145@yahoo.com
YAHOO_APP_PASSWORD=paste-your-16-char-app-password-here
FAMILY_CONTACTS=chandler
```

The `FAMILY_CONTACTS` field is a comma-separated list of patterns to match against
sender names and addresses. "chandler" will match any email where the sender's name
or address contains "chandler".

## 3. Add to Claude Code

Register the MCP server using the CLI (run from any directory):

```
claude mcp add yahoo-mail -- node D:\workspace\yahoomailMCP\server.mjs
```

To add it globally (available in all projects), use the `--scope user` flag:

```
claude mcp add --scope user yahoo-mail -- node D:\workspace\yahoomailMCP\server.mjs
```

Verify it's connected:

```
claude mcp list
```

**Note:** The server reads credentials from the `.env` file in this directory, not from the MCP config. Make sure `.env` is configured before starting Claude Code.

## Available Tools

| Tool | Description |
|------|-------------|
| `check_inbox` | Overview of recent inbox messages with family email highlighting |
| `get_family_emails` | Fetch and preview emails from Chandler family members |
| `read_email` | Read the full content of a specific email |
| `scan_spam` | Scan inbox for likely spam using pattern detection |
| `move_to_junk` | Move confirmed spam to Yahoo's Bulk Mail folder |
| `move_to_inbox` | Rescue false positives from Bulk Mail |
| `search_emails` | Search by sender, subject, date range |
| `list_mailboxes` | List all Yahoo Mail folders |

## Typical Workflow

1. **"Check my inbox"** → runs `check_inbox`, shows summary with [FAMILY] tags
2. **"Any new family emails?"** → runs `get_family_emails` with unread_only
3. **"Scan for spam"** → runs `scan_spam`, shows suspects with scores
4. **"Move those to junk"** → you confirm, then `move_to_junk` runs
