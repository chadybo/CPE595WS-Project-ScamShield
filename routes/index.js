// Mounts all route files onto the Express app
import predictRoute from "./predict.js";

const constructorMethod = (app) => {
    app.use("/predict", predictRoute);

  // Home page
  app.get("/", async (req, res) => {
    try {
      return res.render("home", { title: "Home" });
    } catch (e) {
      return res.render("home");
    }
  });

  // Catch-all 404
  app.use(/(.*)/, (req, res) => {
    return res.status(404).render("error", { title: "Page not found" });
  });
};

export default constructorMethod;
