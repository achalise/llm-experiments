import express from 'express';
import { chatHandler } from './handlers/chatHandler';

const app = express();

app.get('/chat', chatHandler);

export default app;
