import "dotenv/config";
import express from "express";
import { create } from "express-handlebars";
import configRoutes from "./routes/index.js";

const app = express();
const port = process.env.PORT || 3000;

const hbs = create({
  defaultLayout: "main",
});

app.engine("handlebars", hbs.engine);
app.set("view engine", "handlebars");
app.set("views", "./views");

// Parse request bodies
app.use(express.json({ limit: "50kb" }));
app.use(express.urlencoded({ extended: false, limit: "50kb" }));

// Serve static files
app.use("/public", express.static("public"));

configRoutes(app);

app.listen(port, () => {
  console.log(`ScamShield running on http://localhost:${port}`);
});
