import { app } from "../../scripts/app.js";

const NODE_TYPE = "WinnouganCheckpointLoader";

app.registerExtension({
    name: "Winnougan.CheckpointLoader",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.call(this);
            this.color   = "#1a2a1a";
            this.bgcolor = "#0f2a0f";
            this.title   = "👉👈 Winnougan Checkpoint Loader";
        };

        // ── Breathing glow ────────────────────────────────────────────────────
        const origBg = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            origBg?.call(this, ctx);
            if (this.flags?.collapsed) return;
            const w = this.size[0], h = this.size[1] + LiteGraph.NODE_TITLE_HEIGHT;
            const yOff = -LiteGraph.NODE_TITLE_HEIGHT, r = 8;
            const t = Date.now()/1000;
            const pulse  = 0.5+0.5*Math.sin(t*(2*Math.PI/3));
            const pulse2 = 0.5+0.5*Math.sin(t*(2*Math.PI/5)+1.0);
            app.graph.setDirtyCanvas(true, false);
            ctx.save();
            ctx.shadowColor="#22dd66"; ctx.shadowBlur=28+pulse*30;
            ctx.strokeStyle="#22dd66"; ctx.lineWidth=1;
            ctx.globalAlpha=0.12+pulse*0.15;
            ctx.beginPath(); ctx.roundRect(-2,yOff-2,w+4,h+4,r+2); ctx.stroke();
            ctx.shadowColor="#4ade80"; ctx.shadowBlur=18+pulse*22;
            ctx.strokeStyle="#4ade80"; ctx.lineWidth=2;
            ctx.globalAlpha=0.30+pulse*0.40;
            ctx.beginPath(); ctx.roundRect(0,yOff,w,h,r); ctx.stroke();
            ctx.shadowBlur=8+pulse2*10; ctx.globalAlpha=0.55+pulse2*0.35;
            ctx.lineWidth=1.5; ctx.strokeStyle="#6aefa0";
            ctx.beginPath(); ctx.roundRect(1,yOff+1,w-2,h-2,r); ctx.stroke();
            ctx.restore();
        };

        // ── Foreground: badge + status pills ─────────────────────────────────
        const origFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            origFg?.call(this, ctx);
            if (this.flags?.collapsed) return;

            const W   = this.size[0];
            const getW = (name) => this.widgets?.find(ww => ww.name === name);
            const sage   = getW("sage_attention")?.value ?? false;
            const triton = getW("triton")?.value ?? false;

            ctx.save();

            // Badge
            ctx.font="bold 10px sans-serif"; ctx.textAlign="right";
            ctx.textBaseline="alphabetic"; ctx.fillStyle="#4ade80";
            ctx.shadowColor="#4ade80"; ctx.shadowBlur=6;
            ctx.fillText("⚡ WINNOUGAN", W-8, 14);
            ctx.shadowBlur=0; ctx.shadowColor="transparent";

            // Status pills for sage + triton
            const drawPill = (label, color, x, y) => {
                ctx.font="bold 9px monospace";
                const tw = ctx.measureText(label).width;
                const pw = tw+10, ph = 14;
                ctx.beginPath(); ctx.roundRect(x,y,pw,ph,4);
                ctx.fillStyle=color+"22"; ctx.fill();
                ctx.strokeStyle=color; ctx.lineWidth=1; ctx.stroke();
                ctx.fillStyle=color; ctx.textAlign="center";
                ctx.textBaseline="middle";
                ctx.fillText(label, x+pw/2, y+ph/2);
            };

            const TH  = LiteGraph.NODE_TITLE_HEIGHT;
            const wH  = LiteGraph.NODE_WIDGET_HEIGHT ?? 20;
            // Pills float below the title
            let px = 8;
            if (sage)   { drawPill("SAGE",   "#4ade80", px, TH+2); px += ctx.measureText("SAGE").width+22; }
            if (triton) { drawPill("TRITON",  "#7aafff", px, TH+2); }

            ctx.restore();
        };

        nodeType.prototype.computeSize = function () {
            const nWidgets = this.widgets?.length ?? 3;
            const wH = LiteGraph.NODE_WIDGET_HEIGHT ?? 20;
            const TH = LiteGraph.NODE_TITLE_HEIGHT;
            return [340, TH + 8 + nWidgets*(wH+4) + 12];
        };
    },
});
