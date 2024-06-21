import { SqlToolkit, createSqlAgent } from "langchain/agents/toolkits/sql";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { model } from "../openai/model";
import { AgentExecutor } from "langchain/dist/agents/executor";


const datasource = new DataSource({
    type: "postgres",
    host: "localhost",
    port: 5432,
    username: "admin",
    password: "welcome",
    database: "demo_db",
});

export const sqlAgent = async (): Promise<AgentExecutor> => {
    const db = await SqlDatabase.fromDataSourceParams({
        appDataSource: datasource,
    });

    const toolkit = new SqlToolkit(db, model);

    const executor = createSqlAgent(model, toolkit);
    return executor;
}
