# Session Context

## User Prompts

### Prompt 1

Call agents team to refactor our game like this. I see the demo much bug dont like this code so please /scout and do again import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FilesetResolver, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import { Loader2, RotateCcw, Trophy, Hand, Timer, ListOrdered, ArrowRight, User, Star, Wifi, WifiOff, Globe, Copy, Info } from 'lucide-react';

// --- FIREBASE IMPORTS ---
import { initializeApp } from 'firebase/app...

### Prompt 2

## Purpose

Search the codebase for files needed to complete the task using a fast, token efficient agent.

## Variables

USER_PROMPT: directory
SCALE: - (defaults to 3)
REPORT_OUTPUT_DIR: `plans/<plan-name>/reports/scout-report.md`

## Workflow:

- Write a prompt for 'SCALE' number of agents to the `Task` tool that will immediately call the `Bash` tool to run these commands to kick off your agents to conduct the search: spawn many `Explore` subagents to search the codebase in parallel based on ...

