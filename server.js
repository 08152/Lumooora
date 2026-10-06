const express = require("express");
const path = require("path");
const { spawn } = require("child_process");
const crypto = require("crypto");

const {
    normalize,
    tokenize,
    words
} = require("./tokenizer");

const app = express();

const PORT =
    Number(process.env.PORT || 10000);

const PYTHON =
    process.env.PYTHON_BIN || "python3";

const WORKER_FILE =
    path.join(
        __dirname,
        "inference_worker.py"
    );

app.use(
    express.json({
        limit: "10mb"
    })
);


/* =========================================================
   PYTHON WORKER
========================================================= */

let worker = null;

let workerBuffer = "";

const pendingRequests =
    new Map();


function startWorker() {

    console.log(
        "[LUMORA] Starte Python-Worker..."
    );


    worker = spawn(
        PYTHON,
        [
            WORKER_FILE
        ],
        {
            cwd: __dirname,

            stdio: [
                "pipe",
                "pipe",
                "pipe"
            ]
        }
    );


    worker.stdout.setEncoding(
        "utf8"
    );


    worker.stdout.on(
        "data",
        data => {

            workerBuffer += data;


            const lines =
                workerBuffer.split("\n");


            workerBuffer =
                lines.pop() || "";


            for (const line of lines) {

                if (!line.trim()) {
                    continue;
                }


                let result;


                try {

                    result =
                        JSON.parse(line);

                } catch (error) {

                    console.error(
                        "[WORKER] Ungültige Antwort:",
                        line
                    );

                    continue;
                }


                const id =
                    result.id;


                const pending =
                    pendingRequests.get(
                        id
                    );


                if (!pending) {
                    continue;
                }


                pendingRequests.delete(
                    id
                );


                clearTimeout(
                    pending.timeout
                );


                if (result.ok) {

                    pending.resolve(
                        result
                    );

                } else {

                    pending.reject(
                        new Error(
                            result.error ||
                            "Worker-Fehler"
                        )
                    );
                }
            }
        }
    );


    worker.stderr.setEncoding(
        "utf8"
    );


    worker.stderr.on(
        "data",
        data => {

            console.error(
                "[WORKER]",
                data.trim()
            );
        }
    );


    worker.on(
        "error",
        error => {

            console.error(
                "[WORKER ERROR]",
                error.message
            );
        }
    );


    worker.on(
        "exit",
        (code, signal) => {

            console.error(
                "[WORKER EXIT]",
                {
                    code,
                    signal
                }
            );


            for (
                const [
                    id,
                    pending
                ]
                of pendingRequests
            ) {

                clearTimeout(
                    pending.timeout
                );

                pending.reject(
                    new Error(
                        "Python-Worker wurde beendet."
                    )
                );

                pendingRequests.delete(
                    id
                );
            }


            worker = null;


            /*
                Nicht in einer engen Endlosschleife
                neu starten.
            */

            setTimeout(
                () => {

                    if (!worker) {
                        startWorker();
                    }

                },
                3000
            );
        }
    );
}


function sendToWorker(
    action,
    data = {},
    timeoutMs = 120000
) {

    return new Promise(
        (resolve, reject) => {

            if (!worker) {

                return reject(
                    new Error(
                        "KI-Worker ist noch nicht bereit."
                    )
                );
            }


            const id =
                crypto.randomUUID();


            const request = {

                id,

                action,

                ...data

            };


            const timeout =
                setTimeout(
                    () => {

                        pendingRequests.delete(
                            id
                        );

                        reject(
                            new Error(
                                "KI-Anfrage hat das Zeitlimit überschritten."
                            )
                        );

                    },
                    timeoutMs
                );


            pendingRequests.set(
                id,
                {
                    resolve,
                    reject,
                    timeout
                }
            );


            try {

                worker.stdin.write(
                    JSON.stringify(
                        request
                    ) + "\n"
                );

            } catch (error) {

                clearTimeout(
                    timeout
                );

                pendingRequests.delete(
                    id
                );

                reject(error);
            }
        }
    );
}


/* =========================================================
   START
========================================================= */

startWorker();


/* =========================================================
   FRONTEND
========================================================= */

app.get(
    "/",
    (req, res) => {

        res.sendFile(
            path.join(
                __dirname,
                "index.html"
            )
        );
    }
);


/* =========================================================
   CHAT
========================================================= */

app.post(
    "/api/chat",
    async (req, res) => {

        try {

            const message =
                String(
                    req.body?.message ||
                    ""
                ).trim();


            if (!message) {

                return res.status(
                    400
                ).json({

                    error:
                        "Keine Nachricht."

                });
            }


            const result =
                await sendToWorker(
                    "generate",
                    {

                        text:
                            message,

                        max_new_tokens:
                            Number(
                                req.body?.max_new_tokens ||
                                150
                            ),

                        temperature:
                            Number(
                                req.body?.temperature ??
                                0.8
                            ),

                        top_k:
                            Number(
                                req.body?.top_k ||
                                40
                            ),

                        top_p:
                            Number(
                                req.body?.top_p ??
                                0.9
                            )

                    },
                    180000
                );


            res.json({

                answer:
                    result.text,

                model:
                    "Lumora-750M",

                tokens:
                    tokenize(message),

                tokenCount:
                    tokenize(message).length,

                generated:
                    true

            });

        } catch (error) {

            console.error(
                "[CHAT]",
                error
            );


            res.status(
                503
            ).json({

                error:
                    error.message

            });
        }
    }
);


/* =========================================================
   TOKENIZER
========================================================= */

app.post(
    "/api/tokenize",
    (req, res) => {

        const text =
            String(
                req.body?.text ||
                ""
            );


        const result =
            tokenize(text);


        res.json({

            text,

            tokens:
                result,

            count:
                result.length

        });
    }
);


/* =========================================================
   MODEL STATUS
========================================================= */

app.get(
    "/api/model",
    async (req, res) => {

        try {

            const result =
                await sendToWorker(
                    "health",
                    {},
                    30000
                );


            res.json({

                online:
                    true,

                name:
                    result.model,

                device:
                    result.device,

                parameters:
                    result.parameters,

                parametersMillions:
                    result.parameters /
                    1000000,

                trained:
                    true

            });

        } catch (error) {

            res.status(
                503
            ).json({

                online:
                    false,

                error:
                    error.message

            });
        }
    }
);


/* =========================================================
   HEALTH
========================================================= */

app.get(
    "/api/health",
    async (req, res) => {

        try {

            const result =
                await sendToWorker(
                    "health",
                    {},
                    30000
                );


            res.json({

                online:
                    true,

                lumora:
                    true,

                tokenizer:
                    true,

                transformer:
                    true,

                model:
                    result.model,

                device:
                    result.device,

                parameters:
                    result.parameters,

                trainedWeights:
                    true

            });

        } catch (error) {

            res.status(
                503
            ).json({

                online:
                    false,

                lumora:
                    true,

                worker:
                    false,

                error:
                    error.message

            });
        }
    }
);


/* =========================================================
   API 404
========================================================= */

app.use(
    "/api",
    (req, res) => {

        res.status(
            404
        ).json({

            error:
                "API-Endpunkt nicht gefunden."

        });
    }
);


/* =========================================================
   FRONTEND FALLBACK
========================================================= */

app.get(
    "*",
    (req, res) => {

        res.sendFile(
            path.join(
                __dirname,
                "index.html"
            )
        );
    }
);


/* =========================================================
   START SERVER
========================================================= */

app.listen(
    PORT,
    "0.0.0.0",
    () => {

        console.log("");
        console.log(
            "======================================"
        );

        console.log(
            "       LUMORA 750M ONLINE"
        );

        console.log(
            "======================================"
        );

        console.log(
            "Port:",
            PORT
        );

        console.log(
            "Tokenizer: AKTIV"
        );

        console.log(
            "Transformer: AKTIV"
        );

        console.log(
            "Python Worker: AKTIV"
        );

        console.log(
            "======================================"
        );

        console.log("");
    }
);
