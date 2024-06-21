# LangChain SQL Agent

A simple set up to query a Postgres database using natural language.

### Prerequisites

- Docker Desktop should be running
- create a folder `/tmp/postgres/data` for the data volume used by the postgres container
- register for `OpenAI` api key and expose it environment variable:
  `export OPENAI_API_KEY=your_open_ai_api_key` 

### Running the example

- checkout the project and `cd` to the project folder
- `docker-compose up -d`
- execute `schema.sql` against the local postgresse database
- create test data `data.sql`
- `npm install`
- `npm run start:dev`

### Testing

Example using httpie(https://httpie.io/)


```
http http://localhost:3000/chat\?query\=Please+list+the+claim+with+highest+amount
HTTP/1.1 200 OK
Connection: keep-alive
Content-Length: 125
Content-Type: text/html; charset=utf-8
Date: Fri, 21 Jun 2024 19:43:11 GMT
ETag: W/"7d-BBIlPPNC9usHWuIJ7ADTL/gKgIs"
Keep-Alive: timeout=5
X-Powered-By: Express

The claim with the highest amount is claim with ID 5, amount of 300, description of floods, type FLOODS, and status Approved.
```
