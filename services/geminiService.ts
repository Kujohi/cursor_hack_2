import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration } from "@google/genai";
import { arrayBufferToBase64, decodeAudioData, float32ToInt16PCM, base64ToUint8Array } from "./audioUtils";
import { GeoLocation } from "../types";

const API_KEY = process.env.API_KEY || '';
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';

// Define the function tool
const reportEmergencyTool: FunctionDeclaration = {
  name: 'reportEmergency',
  description: 'Create an emergency ticket. Use this ONLY after you have gathered all necessary information (Location, Type, Description, People Count, and Critical Needs).',
  parameters: {
    type: Type.OBJECT,
    properties: {
      emergencyType: {
        type: Type.STRING,
        description: 'Type of disaster (e.g., Flood, Fire, Injury, Trapped, Storm).'
      },
      description: {
        type: Type.STRING,
        description: 'A concise summary of the situation.'
      },
      peopleCount: {
        type: Type.NUMBER,
        description: 'Estimated number of people involved.'
      },
      criticalNeeds: {
        type: Type.STRING,
        description: 'Specific assistance needed (e.g., Water, Boat, Medical).'
      },
      locationName: {
        type: Type.STRING,
        description: 'The specific address, city, or place name if provided by user.'
      },
      latitude: {
        type: Type.NUMBER,
        description: 'The estimated latitude of the locationName. REQUIRED if locationName is provided (unless it is Current Location).'
      },
      longitude: {
        type: Type.NUMBER,
        description: 'The estimated longitude of the locationName. REQUIRED if locationName is provided (unless it is Current Location).'
      }
    },
    required: ['emergencyType', 'description']
  }
};

export class GeminiLiveService {
  private ai: GoogleGenAI;
  private inputAudioContext: AudioContext | null = null;
  private outputAudioContext: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private nextStartTime = 0;
  private sessionPromise: Promise<any> | null = null;
  private currentSession: any = null; // Stored reference to session for closing
  
  public onReportSubmitted: ((report: any) => void) | null = null;
  public onAudioLevel: ((level: number) => void) | null = null;
  public onStatusChange: ((status: string) => void) | null = null;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: API_KEY });
  }

  async connect(userLocation: GeoLocation | null) {
    this.onStatusChange?.('Initializing Audio...');
    
    this.inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
    this.outputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    
    // Resume audio contexts if suspended
    if (this.inputAudioContext.state === 'suspended') await this.inputAudioContext.resume();
    if (this.outputAudioContext.state === 'suspended') await this.outputAudioContext.resume();

    this.onStatusChange?.('Requesting Mic...');
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    this.onStatusChange?.('Connecting to HQ...');
    
    // Format location for system instruction
    const locContext = userLocation 
      ? `DEVICE_GPS_COORDINATES: ${userLocation.lat.toFixed(6)}, ${userLocation.lng.toFixed(6)}`
      : "DEVICE_GPS_COORDINATES: Unavailable (User must provide voice location)";

    this.sessionPromise = this.ai.live.connect({
      model: MODEL_NAME,
      callbacks: {
        onopen: this.handleOpen.bind(this),
        onmessage: this.handleMessage.bind(this),
        onclose: () => {
             this.onStatusChange?.('Disconnected');
             this.stop();
        },
        onerror: (err) => {
            console.error('Live API Error:', err);
            this.onStatusChange?.('Connection Error');
        },
      },
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
        },
        systemInstruction: `You are an emergency response dispatcher for "Disaster Connect". 
        
        CONTEXT:
        ${locContext}
        
        PROTOCOL (Strictly follow this order):
        1. LOCATION: Determine if the user is at their "Current Location" or a remote location.
        2. SITUATION: Ask for the type of emergency and a description of what is happening.
        3. VITAL DETAILS (MANDATORY): You MUST ask specifically for:
           - The number of people involved/trapped.
           - Any critical needs (injuries, medical aid, food, water).
           DO NOT create the ticket yet.
        4. EXECUTION:
           - Only AFTER gathering Location, Situation, People Count, and Needs:
           - Say: "Please hold on, I am submitting the full report now."
           - Call the 'reportEmergency' tool.
        5. CLOSING:
           - Wait for tool completion.
           - Say: "Report submitted. Responders have been notified with all details."
        
        TOOL RULES:
        - If user says "Current Location" or "Here": LEAVE lat/long empty in the tool.
        - If user says a specific place (e.g. "Central Park", "123 Main St"): You MUST ESTIMATE the numeric latitude and longitude for that place and fill them in the tool arguments. Do not leave them empty.
        
        Keep responses calm and professional.`,
        tools: [{ functionDeclarations: [reportEmergencyTool] }]
      }
    });

    this.currentSession = await this.sessionPromise;
  }

  private handleOpen() {
    this.onStatusChange?.('Live');
    if (!this.inputAudioContext || !this.stream) return;

    this.source = this.inputAudioContext.createMediaStreamSource(this.stream);
    this.processor = this.inputAudioContext.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      
      let sum = 0;
      for (let i = 0; i < inputData.length; i++) {
        sum += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sum / inputData.length);
      this.onAudioLevel?.(rms);

      const pcmData = float32ToInt16PCM(inputData);
      const uint8Params = new Uint8Array(pcmData.buffer);
      const base64Data = arrayBufferToBase64(uint8Params.buffer);

      if (this.currentSession) {
        this.currentSession.sendRealtimeInput({
          media: {
            mimeType: 'audio/pcm;rate=16000',
            data: base64Data
          }
        });
      }
    };

    this.source.connect(this.processor);
    this.processor.connect(this.inputAudioContext.destination);
  }

  private async handleMessage(message: LiveServerMessage) {
    // Handle Audio Output
    const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
    if (base64Audio && this.outputAudioContext) {
      const audioData = base64ToUint8Array(base64Audio);
      this.nextStartTime = Math.max(this.nextStartTime, this.outputAudioContext.currentTime);
      
      try {
        const buffer = await decodeAudioData(audioData, this.outputAudioContext);
        const source = this.outputAudioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.outputAudioContext.destination);
        source.start(this.nextStartTime);
        this.nextStartTime += buffer.duration;
      } catch (e) {
        console.error("Error decoding audio", e);
      }
    }

    // Handle Tool Calls
    if (message.toolCall) {
      const responses = [];
      for (const fc of message.toolCall.functionCalls) {
        if (fc.name === 'reportEmergency') {
          // Trigger the frontend callback
          this.onReportSubmitted?.(fc.args);
          responses.push({
            id: fc.id,
            name: fc.name,
            response: { result: { status: 'success', ticketId: Date.now().toString() } }
          });
        }
      }
      
      if (responses.length > 0 && this.currentSession) {
         this.currentSession.sendToolResponse({
           functionResponses: responses
         });
      }
    }
  }

  stop() {
    if (this.processor) {
      this.processor.disconnect();
      this.processor.onaudioprocess = null;
    }
    if (this.source) {
      this.source.disconnect();
    }
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }
    if (this.inputAudioContext) {
      this.inputAudioContext.close();
    }
    if (this.outputAudioContext) {
      this.outputAudioContext.close();
    }
    
    this.processor = null;
    this.source = null;
    this.stream = null;
    this.inputAudioContext = null;
    this.outputAudioContext = null;
    this.currentSession = null;
    this.onStatusChange?.('Disconnected');
  }
}