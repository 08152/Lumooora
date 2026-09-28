// model.js

function generateReply(input) {
  const text = input.toLowerCase();

  if (text.includes("hallo") || text.includes("hi")) {
    return "Hey! Schön, dass du da bist 😄";
  }

  if (text.includes("hilfe")) {
    return "Wobei brauchst du Hilfe?";
  }

  if (text.includes("github")) {
    return "Du kannst diese KI direkt mit GitHub Pages hosten.";
  }

  return "Interessant! Erzähl mir mehr.";
}
