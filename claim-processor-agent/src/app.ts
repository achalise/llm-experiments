import express, { Express, Request, Response } from 'express';
import { graph } from './agent/graph.js';
import { BaseMessage, HumanMessage } from '@langchain/core/messages';

const config = { configurable: { thread_id: `thread-${Math.random()}` } };

const app: Express = express();
//please create an express server and implement the routes as per your requirements

app.get('/api/chat', async (req: Request, res: Response) => {
    const { message } = req.query;
    console.log(`Received message: ${message}`);
    
    const result = await graph.withConfig(config).invoke({
        messages: [new HumanMessage(message as string)],
      },
      { recursionLimit: 15, ...config}); 
    result.messages.forEach((msg: BaseMessage) => console.log(msg.content));
    res.json({ message: result.messages[result.messages.length - 1].content });
});

app.listen(3000, () => {
    console.log('Server is running on port 3000');
});

