// Copyright © 2025 Apple Inc.

public enum Chat {
    public struct Message {
        /// The role of the message sender.
        public var role: Role

        /// The content of the message.
        public var content: String

        /// Array of image data associated with the message.
        public var images: [UserInput.Image]

        /// Array of video data associated with the message.
        public var videos: [UserInput.Video]

        /// For `role: .tool` messages: the id of the preceding tool call this result
        /// corresponds to. Rendered as `tool_call_id` in the message dict, matching the
        /// OpenAI / Anthropic / Gemma 4 wire format so chat templates can correlate
        /// tool responses with calls (`message.tool_call_id == assistant.tool_calls[*].id`).
        public var toolCallId: String?

        /// For `role: .tool` messages: the name of the tool that produced this result.
        /// Rendered as `name` in the message dict. Some templates (notably Gemma 4's
        /// `<|tool_response>response:<name>{...}<tool_response|>`) require this to
        /// render a correlated tool-response turn.
        public var name: String?

        /// For `role: .assistant` messages: tool calls the assistant issued in this turn.
        /// Each entry follows the OpenAI shape
        /// `{"id": String, "type": "function", "function": {"name": String, "arguments": <dict or JSON string>}}`.
        /// Rendered as `tool_calls` in the message dict so chat templates (Gemma 4's
        /// `<|tool_call>call:<name>{...}<tool_call|>`, Qwen's `<tool_call><function=..>..</function></tool_call>`)
        /// can reproduce the assistant's prior call alongside its content.
        public var toolCalls: [[String: any Sendable]]?

        /// For `role: .assistant` messages: the model's prior reasoning / chain-of-thought,
        /// rendered as `reasoning_content` in the message dict. Qwen3-family chat
        /// templates read this field and re-render the prior `<think>...</think>`
        /// content into the prompt during multi-step tool loops, so dropping it
        /// degrades the model's view of its own reasoning history.
        public var reasoningContent: String?

        public init(
            role: Role, content: String, images: [UserInput.Image] = [],
            videos: [UserInput.Video] = [],
            toolCallId: String? = nil, name: String? = nil,
            toolCalls: [[String: any Sendable]]? = nil,
            reasoningContent: String? = nil
        ) {
            self.role = role
            self.content = content
            self.images = images
            self.videos = videos
            self.toolCallId = toolCallId
            self.name = name
            self.toolCalls = toolCalls
            self.reasoningContent = reasoningContent
        }

        public static func system(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            Self(role: .system, content: content, images: images, videos: videos)
        }

        public static func assistant(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = [],
            toolCalls: [[String: any Sendable]]? = nil,
            reasoningContent: String? = nil
        ) -> Self {
            Self(
                role: .assistant, content: content, images: images, videos: videos,
                toolCalls: toolCalls, reasoningContent: reasoningContent
            )
        }

        public static func user(
            _ content: String, images: [UserInput.Image] = [], videos: [UserInput.Video] = []
        ) -> Self {
            Self(role: .user, content: content, images: images, videos: videos)
        }

        public static func tool(
            _ content: String, toolCallId: String? = nil, name: String? = nil
        ) -> Self {
            Self(role: .tool, content: content, toolCallId: toolCallId, name: name)
        }

        public enum Role: String, Sendable {
            case user
            case assistant
            case system
            case tool
        }
    }
}

/// Protocol for something that can convert structured
/// ``Chat/Message`` into model specific ``Message``
/// (raw dictionary) format.
///
/// Typically this is owned and used by a ``UserInputProcessor``:
///
/// ```swift
/// public func prepare(input: UserInput) async throws -> LMInput {
///     let messages = Qwen2VLMessageGenerator().generate(from: input)
///     ...
/// ```
public protocol MessageGenerator: Sendable {

    /// Generates messages from the input.
    func generate(from input: UserInput) -> [Message]

    /// Returns array of `[String: any Sendable]` aka ``Message``
    func generate(messages: [Chat.Message]) -> [Message]

    /// Returns `[String: any Sendable]`, aka ``Message``.
    func generate(message: Chat.Message) -> Message
}

extension MessageGenerator {

    public func generate(message: Chat.Message) -> Message {
        var dict: Message = [
            "role": message.role.rawValue,
            "content": message.content,
        ]
        if let toolCallId = message.toolCallId {
            dict["tool_call_id"] = toolCallId
        }
        if let name = message.name {
            dict["name"] = name
        }
        if let toolCalls = message.toolCalls {
            dict["tool_calls"] = toolCalls
        }
        if let reasoningContent = message.reasoningContent {
            dict["reasoning_content"] = reasoningContent
        }
        return dict
    }

    public func generate(messages: [Chat.Message]) -> [Message] {
        var rawMessages: [Message] = []

        for message in messages {
            let raw = generate(message: message)
            rawMessages.append(raw)
        }

        return rawMessages
    }

    public func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            generate(messages: [.user(text)])
        case .messages(let messages):
            messages
        case .chat(let messages):
            generate(messages: messages)
        }
    }
}

/// Default implementation of ``MessageGenerator`` that produces a
/// `role` and `content`.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct DefaultMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        var dict: Message = [
            "role": message.role.rawValue,
            "content": message.content,
        ]
        if let toolCallId = message.toolCallId {
            dict["tool_call_id"] = toolCallId
        }
        if let name = message.name {
            dict["name"] = name
        }
        if let toolCalls = message.toolCalls {
            dict["tool_calls"] = toolCalls
        }
        if let reasoningContent = message.reasoningContent {
            dict["reasoning_content"] = reasoningContent
        }
        return dict
    }
}

/// Implementation of ``MessageGenerator`` that produces a
/// `role` and `content` but omits `system` roles.
///
/// ```swift
/// [
///     "role": message.role.rawValue,
///     "content": message.content,
/// ]
/// ```
public struct NoSystemMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(messages: [Chat.Message]) -> [Message] {
        messages
            .filter { $0.role != .system }
            .map { generate(message: $0) }
    }
}
