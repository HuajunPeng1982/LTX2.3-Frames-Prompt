import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "LTX2.3.FramesPrompt",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LTX23FramesPrompt") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            // Remove default output widgets so we replace them with styled ones
            if (this.widgets) {
                for (let i = this.widgets.length - 1; i >= 0; i--) {
                    const w = this.widgets[i];
                    if (w.name === "prompts" || w.name === "status") {
                        this.widgets.splice(i, 1);
                    }
                }
            }

            // Prompts display
            const promptsWidget = ComfyWidgets["STRING"](
                this, "prompts_display",
                ["STRING", { multiline: true, default: "" }],
                app
            );
            promptsWidget.widget.readOnly = true;
            promptsWidget.widget.inputEl.placeholder = "Generated prompts will appear here...";
            promptsWidget.widget.inputEl.style.minHeight = "200px";
            promptsWidget.widget.inputEl.style.fontFamily = "monospace";
            promptsWidget.widget.inputEl.style.fontSize = "12px";

            // Status display
            const statusWidget = ComfyWidgets["STRING"](
                this, "status_display",
                ["STRING", { multiline: true, default: "" }],
                app
            );
            statusWidget.widget.readOnly = true;
            statusWidget.widget.inputEl.placeholder = "Process status will appear here...";
            statusWidget.widget.inputEl.style.minHeight = "120px";
            statusWidget.widget.inputEl.style.fontFamily = "monospace";
            statusWidget.widget.inputEl.style.fontSize = "11px";
            statusWidget.widget.inputEl.style.color = "#888";

            this.promptsDisplay = promptsWidget.widget;
            this.statusDisplay = statusWidget.widget;

            // Ensure node is wide enough
            this.size[0] = Math.max(this.size[0], 620);

            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (message.text && message.text.length >= 1) {
                this.promptsDisplay.value = message.text[0] || "";
            }
            if (message.text && message.text.length >= 2) {
                this.statusDisplay.value = message.text[1] || "";
            }
        };
    },
});
